//! Pure-Rust visual and geometric grounding helpers (`#![forbid(unsafe_code)]`).
//!
//! Provides lossless RGBA PNG decode/encode utilities, calibrated coordinate-ruler
//! rendering, edge-padded centered ROI cropping (eliminating top/left edge bias),
//! pixel-level canvas diff analytics, and structured [`SystemGroundingPack`] prompts.

use std::{collections::HashMap, io::Cursor};

use anyhow::{Context as _, Result, bail};
use serde::{Deserialize, Serialize};

use crate::a11y::{A11yEntry, InteractionRoute, REDACTED_VALUE, RouteDecision};

/// Neutral dark-slate RGBA fill `[30, 32, 38, 255]` used for out-of-bounds ROI padding.
pub const ROI_PADDING_RGBA: [u8; 4] = [30, 32, 38, 255];

/// Maximum supported image edge (`16384 px`) to prevent accidental allocation bombs.
const MAX_IMAGE_EDGE: u32 = 16_384;

/// Decoded 8-bit RGBA image buffer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RgbaImage {
    /// Image width in pixels.
    pub width: u32,
    /// Image height in pixels.
    pub height: u32,
    /// Row-major `RGBA8` pixels (`width * height * 4` bytes).
    pub data: Vec<u8>,
}

/// Decodes an arbitrary PNG (RGBA, RGB, Grayscale, Grayscale+Alpha, Indexed, or 16-bit)
/// into canonical 8-bit `RGBA8` pixel memory.
///
/// Time complexity: $O(W \cdot H)$.
/// Space complexity: $O(W \cdot H)$.
pub fn decode_png_rgba8(png_bytes: &[u8]) -> Result<RgbaImage> {
    if png_bytes.is_empty() {
        bail!("cannot decode empty PNG buffer");
    }
    let mut decoder = png::Decoder::new(Cursor::new(png_bytes));
    decoder.set_transformations(png::Transformations::EXPAND | png::Transformations::STRIP_16);
    let mut reader = decoder.read_info().context("invalid PNG header")?;
    let mut raw = vec![0_u8; reader.output_buffer_size()];
    let info = reader
        .next_frame(&mut raw)
        .context("failed to decode PNG frame")?;
    let width = info.width;
    let height = info.height;
    if width == 0 || height == 0 || width > MAX_IMAGE_EDGE || height > MAX_IMAGE_EDGE {
        bail!("unsupported PNG dimensions {width}x{height}");
    }
    if info.bit_depth != png::BitDepth::Eight {
        bail!(
            "unexpected post-transform PNG bit depth {:?}",
            info.bit_depth
        );
    }

    let pixel_count = (width as usize)
        .checked_mul(height as usize)
        .context("PNG pixel count overflow")?;
    let src = &raw[..info.buffer_size()];

    let rgba = match info.color_type {
        png::ColorType::Rgba => {
            if src.len() != pixel_count * 4 {
                bail!("truncated RGBA PNG frame buffer");
            }
            src.to_vec()
        }
        png::ColorType::Rgb => {
            if src.len() != pixel_count * 3 {
                bail!("truncated RGB PNG frame buffer");
            }
            let mut out = Vec::with_capacity(pixel_count * 4);
            for chunk in src.as_chunks::<3>().0 {
                out.extend_from_slice(&[chunk[0], chunk[1], chunk[2], 255]);
            }
            out
        }
        png::ColorType::Grayscale => {
            if src.len() != pixel_count {
                bail!("truncated Grayscale PNG frame buffer");
            }
            let mut out = Vec::with_capacity(pixel_count * 4);
            for &l in src {
                out.extend_from_slice(&[l, l, l, 255]);
            }
            out
        }
        png::ColorType::GrayscaleAlpha => {
            if src.len() != pixel_count * 2 {
                bail!("truncated GrayscaleAlpha PNG frame buffer");
            }
            let mut out = Vec::with_capacity(pixel_count * 4);
            for chunk in src.as_chunks::<2>().0 {
                let l = chunk[0];
                let a = chunk[1];
                out.extend_from_slice(&[l, l, l, a]);
            }
            out
        }
        png::ColorType::Indexed => {
            bail!("indexed PNG did not expand to RGB/RGBA");
        }
    };

    Ok(RgbaImage {
        width,
        height,
        data: rgba,
    })
}

/// Encodes an `RGBA8` buffer (`width * height * 4` bytes) into a compressed PNG.
///
/// Time complexity: $O(W \cdot H)$.
/// Space complexity: $O(W \cdot H)$.
pub fn encode_rgba8_png(rgba: &[u8], width: u32, height: u32) -> Result<Vec<u8>> {
    if width == 0 || height == 0 || width > MAX_IMAGE_EDGE || height > MAX_IMAGE_EDGE {
        bail!("invalid RGBA dimensions {width}x{height}");
    }
    let expected = (width as usize)
        .checked_mul(height as usize)
        .and_then(|n| n.checked_mul(4))
        .context("RGBA buffer size overflow")?;
    if rgba.len() != expected {
        bail!(
            "RGBA byte length mismatch: expected {expected} for {width}x{height}, got {}",
            rgba.len()
        );
    }

    let mut out = Vec::new();
    {
        let mut encoder = png::Encoder::new(&mut out, width, height);
        encoder.set_color(png::ColorType::Rgba);
        encoder.set_depth(png::BitDepth::Eight);
        let mut writer = encoder.write_header().context("PNG header write failed")?;
        writer
            .write_image_data(rgba)
            .context("PNG image data write failed")?;
    }
    Ok(out)
}

/// Draws calibrated high-contrast coordinate ticks along the top (`x = 50, 100, 150...`)
/// and left (`y = 50, 100, 150...`) margins with legible 3x5 pixel numerals inside compact
/// dark badge plates, plus an unmistakable translucent crosshair at `cursor_xy` when provided.
///
/// Compact edge strips avoid obscuring the center workspace area.
///
/// Time complexity: $O(W \cdot H)$ (full PNG decode + $O(W + H)$ overlay + PNG encode).
/// Space complexity: $O(W \cdot H)$ for the decoded RGBA framebuffer.
pub fn render_calibrated_ruler_png(
    png_bytes: &[u8],
    cursor_xy: Option<(f64, f64)>,
) -> Result<Vec<u8>> {
    let mut img = decode_png_rgba8(png_bytes)?;
    let w = img.width as i32;
    let h = img.height as i32;

    // Subtle 2px perimeter guide strips along top & left so ticks read cleanly against light/dark UIs.
    fill_rect_blend(&mut img, 0, 0, w, 2, [18, 22, 30, 190]);
    fill_rect_blend(&mut img, 0, 0, 2, h, [18, 22, 30, 190]);

    // Origin badge "0" at top-left corner.
    draw_label_badge(&mut img, 2, 2, "0", [255, 232, 140, 255], [16, 18, 24, 220]);

    // Top horizontal ticks every 50px, numeral badges every 100px.
    let mut x = 50_i32;
    while x < w {
        let major = x % 100 == 0;
        let tick_h = if major { 10 } else { 5 };
        draw_vertical_tick(&mut img, x, 0, tick_h);
        if major && x + 16 < w {
            let label = x.to_string();
            draw_label_badge(
                &mut img,
                x + 2,
                2,
                &label,
                [255, 240, 180, 255],
                [16, 18, 24, 220],
            );
        }
        x += 50;
    }

    // Left vertical ticks every 50px, numeral badges every 100px.
    let mut y = 50_i32;
    while y < h {
        let major = y % 100 == 0;
        let tick_w = if major { 10 } else { 5 };
        draw_horizontal_tick(&mut img, 0, y, tick_w);
        if major && y + 10 < h {
            let label = y.to_string();
            draw_label_badge(
                &mut img,
                2,
                y + 2,
                &label,
                [175, 240, 255, 255],
                [16, 18, 24, 220],
            );
        }
        y += 50;
    }

    if let Some((cx, cy)) = cursor_xy
        && cx.is_finite()
        && cy.is_finite()
    {
        draw_translucent_crosshair(
            &mut img,
            cx.round() as i32,
            cy.round() as i32,
            14,
            3,
            [255, 64, 160, 200],
            [18, 20, 26, 165],
        );
    }

    encode_rgba8_png(&img.data, img.width, img.height)
}

/// Centered square ROI crop preserving exact coordinate mapping back to the source frame.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CenteredRoiCrop {
    /// Encoded `crop_size x crop_size` PNG bytes.
    pub png_bytes: Vec<u8>,
    /// Square crop side length in pixels.
    pub crop_size: u32,
    /// Requested target center X coordinate in source frame space.
    pub center_x: f64,
    /// Requested target center Y coordinate in source frame space.
    pub center_y: f64,
    /// Signed X coordinate in source space corresponding to ROI pixel `(0, 0)`.
    pub origin_x: i32,
    /// Signed Y coordinate in source space corresponding to ROI pixel `(0, 0)`.
    pub origin_y: i32,
    /// Source image width in pixels.
    pub source_width: u32,
    /// Source image height in pixels.
    pub source_height: u32,
}

impl CenteredRoiCrop {
    /// Maps an ROI-local coordinate `(roi_x, roi_y)` back to source image coordinates.
    #[must_use]
    pub fn roi_to_source_xy(&self, roi_x: f64, roi_y: f64) -> (f64, f64) {
        (
            f64::from(self.origin_x) + roi_x,
            f64::from(self.origin_y) + roi_y,
        )
    }
}

/// Extracts a `crop_size x crop_size` PNG centered strictly around `(center_x, center_y)`,
/// filling out-of-bounds margins with neutral slate [`ROI_PADDING_RGBA`] (`[30, 32, 38, 255]`)
/// so targets near top/left/bottom/right edges remain centered without edge-clamp drift.
///
/// Draws a subtle open-center crosshair around `(crop_size/2, crop_size/2)` without overwriting
/// the center `5x5` target pixels.
///
/// Time complexity: $O(W \cdot H + \text{crop\_size}^2)$ (dominated by PNG decode/encode; the ROI copy itself is $O(\text{crop\_size}^2)$).
/// Space complexity: $O(W \cdot H + \text{crop\_size}^2)$.
pub fn crop_centered_roi_png(
    png_bytes: &[u8],
    center_x: f64,
    center_y: f64,
    crop_size: u32,
) -> Result<CenteredRoiCrop> {
    if crop_size == 0 || crop_size > MAX_IMAGE_EDGE {
        bail!("invalid ROI crop_size {crop_size}");
    }
    if !center_x.is_finite() || !center_y.is_finite() {
        bail!("ROI center coordinates must be finite");
    }

    let src = decode_png_rgba8(png_bytes)?;
    let half = (crop_size as i32) / 2;
    let origin_x = (center_x.round() as i32) - half;
    let origin_y = (center_y.round() as i32) - half;

    let size_usize = crop_size as usize;
    let mut dst_data = vec![0_u8; size_usize * size_usize * 4];
    for px in dst_data.as_chunks_mut::<4>().0 {
        px.copy_from_slice(&ROI_PADDING_RGBA);
    }

    let src_w = src.width as i32;
    let src_h = src.height as i32;
    for dy in 0..size_usize {
        let sy = origin_y + dy as i32;
        if sy < 0 || sy >= src_h {
            continue;
        }
        for dx in 0..size_usize {
            let sx = origin_x + dx as i32;
            if sx < 0 || sx >= src_w {
                continue;
            }
            let s_off = ((sy as usize) * (src.width as usize) + (sx as usize)) * 4;
            let d_off = (dy * size_usize + dx) * 4;
            dst_data[d_off..d_off + 4].copy_from_slice(&src.data[s_off..s_off + 4]);
        }
    }

    let mut roi_img = RgbaImage {
        width: crop_size,
        height: crop_size,
        data: dst_data,
    };
    draw_translucent_crosshair(
        &mut roi_img,
        half,
        half,
        10,
        3,
        [100, 225, 255, 110],
        [16, 18, 24, 90],
    );

    let out_png = encode_rgba8_png(&roi_img.data, crop_size, crop_size)?;
    Ok(CenteredRoiCrop {
        png_bytes: out_png,
        crop_size,
        center_x,
        center_y,
        origin_x,
        origin_y,
        source_width: src.width,
        source_height: src.height,
    })
}

/// Quantitative pixel-change metrics computed between two aligned PNG screenshots.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CanvasDiffMetrics {
    /// Total pixels in the image (`width * height`).
    pub total_pixels: usize,
    /// Count of pixels whose max RGBA channel delta exceeded `channel_threshold`.
    pub changed_pixels: usize,
    /// Fraction of pixels changed (`changed_pixels / total_pixels`).
    pub changed_ratio: f64,
    /// Bounding box `[x, y, width, height]` of all changed pixels, if any.
    pub ink_bbox: Option<[u32; 4]>,
    /// Centroid `[cx, cy]` of changed pixels in image space, if any.
    pub centroid_xy: Option<[f64; 2]>,
    /// Dominant `[r, g, b]` color among changed pixels in the `after` frame, if any.
    pub dominant_rgb: Option<[u8; 3]>,
}

/// Computes exact per-pixel diff geometry and dominant stroke color between `before_png`
/// and `after_png` at `channel_threshold` (`max(|dr|, |dg|, |db|, |da|) > channel_threshold`).
///
/// Time complexity: $O(W \cdot H)$.
/// Space complexity: $O(W \cdot H)$ for decoded framebuffers (quantized histogram is $O(1)$, $\le 4096$ buckets).
pub fn compute_canvas_diff(
    before_png: &[u8],
    after_png: &[u8],
    channel_threshold: u8,
) -> Result<CanvasDiffMetrics> {
    let before = decode_png_rgba8(before_png)?;
    let after = decode_png_rgba8(after_png)?;
    if before.width != after.width || before.height != after.height {
        bail!(
            "cannot diff mismatched frame sizes {}x{} vs {}x{}",
            before.width,
            before.height,
            after.width,
            after.height
        );
    }

    let w = before.width;
    let h = before.height;
    let total_pixels = (w as usize) * (h as usize);
    let mut changed_pixels = 0usize;
    let mut min_x = u32::MAX;
    let mut min_y = u32::MAX;
    let mut max_x = 0u32;
    let mut max_y = 0u32;
    let mut sum_x = 0.0_f64;
    let mut sum_y = 0.0_f64;

    // 4-bit-per-channel color bucket -> (count, sum_r, sum_g, sum_b). At most 4096 entries.
    let mut histogram: HashMap<u16, (u64, u64, u64, u64)> = HashMap::new();

    for y in 0..h {
        for x in 0..w {
            let off = ((y as usize) * (w as usize) + (x as usize)) * 4;
            let b = &before.data[off..off + 4];
            let a = &after.data[off..off + 4];
            let max_delta = b[0]
                .abs_diff(a[0])
                .max(b[1].abs_diff(a[1]))
                .max(b[2].abs_diff(a[2]))
                .max(b[3].abs_diff(a[3]));
            if max_delta > channel_threshold {
                changed_pixels += 1;
                min_x = min_x.min(x);
                min_y = min_y.min(y);
                max_x = max_x.max(x);
                max_y = max_y.max(y);
                sum_x += f64::from(x);
                sum_y += f64::from(y);

                let key = ((u16::from(a[0] >> 4)) << 8)
                    | ((u16::from(a[1] >> 4)) << 4)
                    | u16::from(a[2] >> 4);
                let bucket = histogram.entry(key).or_insert((0, 0, 0, 0));
                bucket.0 += 1;
                bucket.1 += u64::from(a[0]);
                bucket.2 += u64::from(a[1]);
                bucket.3 += u64::from(a[2]);
            }
        }
    }

    let (ink_bbox, centroid_xy, dominant_rgb) = if changed_pixels == 0 {
        (None, None, None)
    } else {
        let bbox = [min_x, min_y, max_x - min_x + 1, max_y - min_y + 1];
        let centroid = [
            sum_x / (changed_pixels as f64),
            sum_y / (changed_pixels as f64),
        ];
        let best = histogram.into_values().max_by_key(|v| v.0);
        let dom =
            best.map(|(cnt, sr, sg, sb)| [(sr / cnt) as u8, (sg / cnt) as u8, (sb / cnt) as u8]);
        (Some(bbox), Some(centroid), dom)
    };

    Ok(CanvasDiffMetrics {
        total_pixels,
        changed_pixels,
        changed_ratio: (changed_pixels as f64) / (total_pixels.max(1) as f64),
        ink_bbox,
        centroid_xy,
        dominant_rgb,
    })
}

/// Complete structured spatial & semantic grounding context bundled for a planner/VLM step.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct SystemGroundingPack {
    /// Situational interaction route selected for the active window & goal.
    pub route: RouteDecision,
    /// Last known cursor coordinate `(x, y)` in window-local pixels.
    pub cursor_xy: Option<(f64, f64)>,
    /// Active window dimensions `(width, height)` in pixels.
    pub active_window_size: (u32, u32),
    /// Optional compositor hit-test / occlusion summary.
    pub hit_test_summary: Option<String>,
    /// Top relevant AT-SPI entries with scoped `@<snap>:eN` refs and rects.
    pub top_a11y_entries: Vec<A11yEntry>,
    /// Recent canvas diff summary produced after the previous stroke/action.
    pub last_diff: Option<CanvasDiffMetrics>,
    /// Active drawing canvas bounds `[x, y, width, height]`, if present.
    pub canvas_bounds: Option<[u32; 4]>,
}

impl SystemGroundingPack {
    /// Formats an unambiguous, model-ready grounding prompt block listing interaction route,
    /// cursor coordinates, occlusion state, active canvas geometry, AT-SPI button centers/rects,
    /// and recent stroke diff metrics.
    ///
    /// Time complexity: $O(E)$ where $E$ is `self.top_a11y_entries.len()`.
    /// Space complexity: $O(E)$ for the rendered prompt string.
    #[must_use]
    pub fn format_prompt_block(&self) -> String {
        let route_name = match self.route.route {
            InteractionRoute::SemanticAtSpi => "semantic_at_spi",
            InteractionRoute::PrecisionCanvas => "precision_canvas",
            InteractionRoute::RelativeGame => "relative_game",
        };
        let mut out = String::new();
        out.push_str("### SYSTEM GROUNDING PACK\n");
        out.push_str(&format!(
            "route={route_name} prefer_roi_crop={} profile={} hz={} reason={:?}\n",
            self.route.prefer_roi_crop,
            self.route.recommended_profile,
            self.route.recommended_hz,
            self.route.reason
        ));
        out.push_str(&format!(
            "window_size={}x{}\n",
            self.active_window_size.0, self.active_window_size.1
        ));
        match self.cursor_xy {
            Some((cx, cy)) => out.push_str(&format!("cursor_xy=({cx:.1}, {cy:.1})\n")),
            None => out.push_str("cursor_xy=unknown\n"),
        }

        let has_modal_entry = self
            .top_a11y_entries
            .iter()
            .any(|e| e.role == "dialog" || e.name.to_ascii_lowercase().contains("dismiss"));
        let occlusion_status = self
            .hit_test_summary
            .as_deref()
            .unwrap_or(if has_modal_entry {
                "occluded_by_modal_dialog"
            } else {
                "clear_unoccluded"
            });
        out.push_str(&format!("occlusion_status={occlusion_status:?}\n"));

        if let Some([x, y, w, h]) = self.canvas_bounds {
            let cx = f64::from(x) + f64::from(w) * 0.5;
            let cy = f64::from(y) + f64::from(h) * 0.5;
            out.push_str(&format!(
                "canvas_bounds=[{x},{y},{w},{h}] canvas_center=({cx:.1}, {cy:.1})\n"
            ));
        }

        if let Some(diff) = &self.last_diff {
            out.push_str(&format!(
                "last_diff: changed_pixels={} changed_ratio={:.4} bbox={:?} centroid={:?} dominant_rgb={:?}\n",
                diff.changed_pixels,
                diff.changed_ratio,
                diff.ink_bbox,
                diff.centroid_xy,
                diff.dominant_rgb
            ));
        }

        if !self.top_a11y_entries.is_empty() {
            out.push_str("ui_elements:\n");
            for entry in &self.top_a11y_entries {
                let rect_part = if let Some([rx, ry, rw, rh]) = entry.rect {
                    let cx = f64::from(rx) + f64::from(rw) * 0.5;
                    let cy = f64::from(ry) + f64::from(rh) * 0.5;
                    format!("rect=[{rx},{ry},{rw},{rh}] center=({cx:.1}, {cy:.1})")
                } else {
                    "rect=none".to_string()
                };
                let val_part = if entry.is_secure {
                    format!(" val={REDACTED_VALUE:?} [secure]")
                } else if let Some(val) = &entry.value {
                    format!(" val={val:?}")
                } else {
                    String::new()
                };
                out.push_str(&format!(
                    "- {} [{}] {:?}{} {} actions=[{}]\n",
                    entry.ref_id,
                    entry.role,
                    entry.name,
                    val_part,
                    rect_part,
                    entry.actions.join(",")
                ));
            }
        }
        out
    }
}

fn blend_pixel(img: &mut RgbaImage, x: i32, y: i32, src: [u8; 4]) {
    if x < 0 || y < 0 || x >= img.width as i32 || y >= img.height as i32 {
        return;
    }
    let off = ((y as usize) * (img.width as usize) + (x as usize)) * 4;
    let a = u16::from(src[3]);
    if a == 255 {
        img.data[off..off + 4].copy_from_slice(&src);
        return;
    }
    if a == 0 {
        return;
    }
    let inv = 255 - a;
    for c in 0..3 {
        let blended = (u16::from(src[c]) * a + u16::from(img.data[off + c]) * inv) / 255;
        img.data[off + c] = blended as u8;
    }
    img.data[off + 3] = 255;
}

fn fill_rect_blend(img: &mut RgbaImage, x: i32, y: i32, w: i32, h: i32, color: [u8; 4]) {
    let x0 = x.max(0);
    let y0 = y.max(0);
    let x1 = (x + w).min(img.width as i32);
    let y1 = (y + h).min(img.height as i32);
    for py in y0..y1 {
        for px in x0..x1 {
            blend_pixel(img, px, py, color);
        }
    }
}

fn draw_vertical_tick(img: &mut RgbaImage, x: i32, y0: i32, len: i32) {
    for py in y0..(y0 + len) {
        blend_pixel(img, x - 1, py, [14, 16, 22, 210]);
        blend_pixel(img, x, py, [250, 250, 255, 255]);
        blend_pixel(img, x + 1, py, [14, 16, 22, 210]);
    }
}

fn draw_horizontal_tick(img: &mut RgbaImage, x0: i32, y: i32, len: i32) {
    for px in x0..(x0 + len) {
        blend_pixel(img, px, y - 1, [14, 16, 22, 210]);
        blend_pixel(img, px, y, [250, 250, 255, 255]);
        blend_pixel(img, px, y + 1, [14, 16, 22, 210]);
    }
}

fn draw_translucent_crosshair(
    img: &mut RgbaImage,
    cx: i32,
    cy: i32,
    arm: i32,
    gap: i32,
    fg: [u8; 4],
    halo: [u8; 4],
) {
    for d in -arm..=arm {
        if d.abs() < gap {
            continue;
        }
        blend_pixel(img, cx + d, cy - 1, halo);
        blend_pixel(img, cx + d, cy + 1, halo);
        blend_pixel(img, cx + d, cy, fg);

        blend_pixel(img, cx - 1, cy + d, halo);
        blend_pixel(img, cx + 1, cy + d, halo);
        blend_pixel(img, cx, cy + d, fg);
    }
}

fn draw_label_badge(img: &mut RgbaImage, x: i32, y: i32, text: &str, fg: [u8; 4], bg: [u8; 4]) {
    if text.is_empty() {
        return;
    }
    let text_w = (text.chars().count() as i32) * 4 - 1;
    fill_rect_blend(img, x, y, text_w + 2, 7, bg);
    draw_text_3x5(
        &mut img.data,
        (img.width, img.height),
        (x + 1, y + 1),
        text,
        1,
        fg,
    );
}

/// Renders uppercase ASCII / digits in a compact 3x5 pixel bitmap font into an `RGBA8` buffer.
pub(crate) fn draw_text_3x5(
    rgba: &mut [u8],
    dims: (u32, u32),
    origin: (i32, i32),
    text: &str,
    scale: i32,
    color: [u8; 4],
) {
    let (img_w, img_h) = dims;
    let (x, y) = origin;
    let mut cursor_x = x;
    let s = scale.max(1);
    for ch in text.chars() {
        let glyph = glyph_3x5(ch.to_ascii_uppercase());
        for (row_idx, row_bits) in glyph.iter().enumerate() {
            for col_idx in 0..3_usize {
                if (row_bits >> (2 - col_idx)) & 1 == 1 {
                    for sy in 0..s {
                        for sx in 0..s {
                            let px = cursor_x + (col_idx as i32) * s + sx;
                            let py = y + (row_idx as i32) * s + sy;
                            if px >= 0 && py >= 0 && px < img_w as i32 && py < img_h as i32 {
                                let off = ((py as usize) * (img_w as usize) + (px as usize)) * 4;
                                if off + 4 <= rgba.len() {
                                    rgba[off..off + 4].copy_from_slice(&color);
                                }
                            }
                        }
                    }
                }
            }
        }
        cursor_x += 4 * s;
    }
}

fn glyph_3x5(ch: char) -> [u8; 5] {
    match ch {
        '0' => [0b111, 0b101, 0b101, 0b101, 0b111],
        '1' => [0b010, 0b110, 0b010, 0b010, 0b111],
        '2' => [0b111, 0b001, 0b111, 0b100, 0b111],
        '3' => [0b111, 0b001, 0b111, 0b001, 0b111],
        '4' => [0b101, 0b101, 0b111, 0b001, 0b001],
        '5' => [0b111, 0b100, 0b111, 0b001, 0b111],
        '6' => [0b111, 0b100, 0b111, 0b101, 0b111],
        '7' => [0b111, 0b001, 0b001, 0b010, 0b010],
        '8' => [0b111, 0b101, 0b111, 0b101, 0b111],
        '9' => [0b111, 0b101, 0b111, 0b001, 0b111],
        'A' => [0b010, 0b101, 0b111, 0b101, 0b101],
        'B' => [0b110, 0b101, 0b110, 0b101, 0b110],
        'C' => [0b111, 0b100, 0b100, 0b100, 0b111],
        'D' => [0b110, 0b101, 0b101, 0b101, 0b110],
        'E' => [0b111, 0b100, 0b110, 0b100, 0b111],
        'F' => [0b111, 0b100, 0b110, 0b100, 0b100],
        'G' => [0b111, 0b100, 0b101, 0b101, 0b111],
        'H' => [0b101, 0b101, 0b111, 0b101, 0b101],
        'I' => [0b111, 0b010, 0b010, 0b010, 0b111],
        'K' => [0b101, 0b101, 0b110, 0b101, 0b101],
        'L' => [0b100, 0b100, 0b100, 0b100, 0b111],
        'M' => [0b101, 0b111, 0b101, 0b101, 0b101],
        'N' => [0b111, 0b101, 0b101, 0b101, 0b101],
        'O' => [0b111, 0b101, 0b101, 0b101, 0b111],
        'P' => [0b111, 0b101, 0b111, 0b100, 0b100],
        'R' => [0b110, 0b101, 0b110, 0b101, 0b101],
        'S' => [0b111, 0b100, 0b111, 0b001, 0b111],
        'T' => [0b111, 0b010, 0b010, 0b010, 0b010],
        'U' => [0b101, 0b101, 0b101, 0b101, 0b111],
        'V' => [0b101, 0b101, 0b101, 0b101, 0b010],
        'W' => [0b101, 0b101, 0b101, 0b111, 0b101],
        'X' => [0b101, 0b101, 0b010, 0b101, 0b101],
        'Y' => [0b101, 0b101, 0b010, 0b010, 0b010],
        ':' => [0b000, 0b010, 0b000, 0b010, 0b000],
        '-' => [0b000, 0b000, 0b111, 0b000, 0b000],
        _ => [0b000, 0b000, 0b000, 0b000, 0b000],
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::a11y::select_interaction_route;

    fn make_solid_png(w: u32, h: u32, color: [u8; 4]) -> Vec<u8> {
        let mut data = Vec::with_capacity((w * h * 4) as usize);
        for _ in 0..(w * h) {
            data.extend_from_slice(&color);
        }
        encode_rgba8_png(&data, w, h).unwrap()
    }

    #[test]
    fn decodes_rgb_grayscale_and_rgba_pngs_cleanly() {
        let mut rgb_bytes = Vec::new();
        {
            let mut enc = png::Encoder::new(&mut rgb_bytes, 4, 4);
            enc.set_color(png::ColorType::Rgb);
            enc.set_depth(png::BitDepth::Eight);
            let mut w = enc.write_header().unwrap();
            w.write_image_data(&[10, 20, 30].repeat(16)).unwrap();
        }
        let img = decode_png_rgba8(&rgb_bytes).unwrap();
        assert_eq!(img.width, 4);
        assert_eq!(img.height, 4);
        assert_eq!(&img.data[0..4], &[10, 20, 30, 255]);

        let mut gray_bytes = Vec::new();
        {
            let mut enc = png::Encoder::new(&mut gray_bytes, 2, 2);
            enc.set_color(png::ColorType::Grayscale);
            enc.set_depth(png::BitDepth::Eight);
            let mut w = enc.write_header().unwrap();
            w.write_image_data(&[99, 99, 99, 99]).unwrap();
        }
        let gimg = decode_png_rgba8(&gray_bytes).unwrap();
        assert_eq!(&gimg.data[0..4], &[99, 99, 99, 255]);
    }

    #[test]
    fn calibrated_ruler_draws_ticks_and_crosshair() {
        let base = make_solid_png(320, 240, [255, 255, 255, 255]);
        let ruled = render_calibrated_ruler_png(&base, Some((160.0, 120.0))).unwrap();
        let diff = compute_canvas_diff(&base, &ruled, 5).unwrap();
        assert!(diff.changed_pixels > 50);
        // Center pixel should remain unobscured because crosshair uses an open center gap.
        let decoded = decode_png_rgba8(&ruled).unwrap();
        let off = (120 * 320 + 160) * 4;
        assert_eq!(&decoded.data[off..off + 4], &[255, 255, 255, 255]);
    }

    #[test]
    fn centered_roi_crop_near_edge_pads_with_slate_and_maps_coordinates_exactly() {
        let w = 400_u32;
        let h = 300_u32;
        let mut rgba = vec![240_u8; (w * h * 4) as usize];
        // Place bright red target pixel at (20, 20).
        let target_off = (20 * (w as usize) + 20) * 4;
        rgba[target_off..target_off + 4].copy_from_slice(&[220, 30, 30, 255]);
        let png = encode_rgba8_png(&rgba, w, h).unwrap();

        let roi = crop_centered_roi_png(&png, 20.0, 20.0, 256).unwrap();
        assert_eq!(roi.crop_size, 256);
        assert_eq!(roi.origin_x, -108);
        assert_eq!(roi.origin_y, -108);
        assert_eq!(roi.roi_to_source_xy(128.0, 128.0), (20.0, 20.0));

        let decoded = decode_png_rgba8(&roi.png_bytes).unwrap();
        // Top-left (0, 0) in ROI maps to (-108, -108) -> neutral slate padding.
        assert_eq!(&decoded.data[0..4], &ROI_PADDING_RGBA);
        // Center (128, 128) in ROI preserves exact red target pixel from (20, 20)!
        let center_off = (128 * 256 + 128) * 4;
        assert_eq!(
            &decoded.data[center_off..center_off + 4],
            &[220, 30, 30, 255]
        );
    }

    #[test]
    fn canvas_diff_computes_exact_bbox_centroid_and_dominant_rgb() {
        let w = 100_u32;
        let h = 80_u32;
        let before = make_solid_png(w, h, [255, 255, 255, 255]);
        let mut after_img = decode_png_rgba8(&before).unwrap();
        for y in 20..30 {
            for x in 40..50 {
                let off = (y * (w as usize) + x) * 4;
                after_img.data[off..off + 4].copy_from_slice(&[50, 185, 70, 255]);
            }
        }
        let after = encode_rgba8_png(&after_img.data, w, h).unwrap();

        let diff = compute_canvas_diff(&before, &after, 10).unwrap();
        assert_eq!(diff.total_pixels, 8000);
        assert_eq!(diff.changed_pixels, 100);
        assert_eq!(diff.ink_bbox, Some([40, 20, 10, 10]));
        let c = diff.centroid_xy.unwrap();
        assert!((c[0] - 44.5).abs() < 1e-6);
        assert!((c[1] - 24.5).abs() < 1e-6);
        assert_eq!(diff.dominant_rgb, Some([50, 185, 70]));
    }

    #[test]
    fn grounding_pack_formats_structured_spatial_prompt_and_redacts_secure_fields() {
        let pack = SystemGroundingPack {
            route: select_interaction_route(
                "virtual_paint",
                "VirtualPaint",
                "paint rainbow",
                5,
                false,
            ),
            cursor_xy: Some((54.0, 36.0)),
            active_window_size: (640, 400),
            hit_test_summary: Some("clear_unoccluded".to_string()),
            top_a11y_entries: vec![
                A11yEntry {
                    ref_id: "@s1:e2".to_string(),
                    raw_ref: "e2".to_string(),
                    role: "push_button".to_string(),
                    name: "RED".to_string(),
                    value: None,
                    rect: Some([16, 16, 76, 40]),
                    actions: vec!["click".to_string()],
                    depth: 1,
                    is_secure: false,
                    child_count: 0,
                },
                A11yEntry {
                    ref_id: "@s1:e3".to_string(),
                    raw_ref: "e3".to_string(),
                    role: "password_text".to_string(),
                    name: "Secret".to_string(),
                    value: Some("should-never-print".to_string()),
                    rect: Some([16, 80, 100, 24]),
                    actions: vec!["set_value".to_string()],
                    depth: 1,
                    is_secure: true,
                    child_count: 0,
                },
            ],
            last_diff: None,
            canvas_bounds: Some([24, 88, 592, 292]),
        };

        let block = pack.format_prompt_block();
        assert!(block.contains("route=precision_canvas"));
        assert!(block.contains("center=(54.0, 36.0)"));
        assert!(block.contains("canvas_bounds=[24,88,592,292]"));
        assert!(block.contains("@s1:e2"));
        assert!(!block.contains("should-never-print"));
        assert!(block.contains(REDACTED_VALUE));
    }

    #[test]
    fn roi_crop_bottom_right_edge_pads_and_diff_rejects_mismatched_dimensions() {
        let w = 200_u32;
        let h = 150_u32;
        let mut rgba = vec![250_u8; (w * h * 4) as usize];
        let br_off = (145 * (w as usize) + 195) * 4;
        rgba[br_off..br_off + 4].copy_from_slice(&[20, 120, 240, 255]);
        let png = encode_rgba8_png(&rgba, w, h).unwrap();

        let roi = crop_centered_roi_png(&png, 195.0, 145.0, 256).unwrap();
        assert_eq!(roi.origin_x, 67);
        assert_eq!(roi.origin_y, 17);
        assert_eq!(roi.roi_to_source_xy(128.0, 128.0), (195.0, 145.0));

        let decoded = decode_png_rgba8(&roi.png_bytes).unwrap();
        // Bottom-right corner of ROI (255, 255) maps to (322, 272) -> out of bounds -> slate padding.
        let oob_off = (255 * 256 + 255) * 4;
        assert_eq!(&decoded.data[oob_off..oob_off + 4], &ROI_PADDING_RGBA);
        let center_off = (128 * 256 + 128) * 4;
        assert_eq!(
            &decoded.data[center_off..center_off + 4],
            &[20, 120, 240, 255]
        );

        let same_diff = compute_canvas_diff(&png, &png, 0).unwrap();
        assert_eq!(same_diff.changed_pixels, 0);
        assert!(same_diff.ink_bbox.is_none());

        assert!(compute_canvas_diff(&png, &roi.png_bytes, 0).is_err());
    }
}
