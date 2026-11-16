-- ~/.config/jiku/config.lua — Sandboxed Lua configuration for jikud.
--
-- Loaded by `jikud --config ~/.config/jiku/config.lua`, by `$JIKU_CONFIG`, or
-- automatically when `~/.config/jiku/config.lua` exists. In the default search it
-- is merged over `~/.config/jiku/config.toml`, so an overlay only states the keys
-- it changes: objects merge key by key, arrays and scalars replace.
--
-- Template values: replace every `<...-model-id>` and `JIKU_*_API_KEY` name with
-- your own provider bindings. `api_key_env` names the environment variable that
-- holds each key; never inline raw secrets here.
--
-- The VM has no `io`, `os`, `package`, `dofile`, or `loadfile`: it cannot read the
-- environment or the filesystem, which keeps secrets out of the Lua heap.

jiku.config{
    daemon = {
        locale = "zh-CN",
        socket_name = "jiku.sock",
        dropbox_dir = "~/JikuBox",
        audit_log = true,
    },

    koma = {
        default_workspace = "ws_01",
        default_overlay = "grid_225",
        perceptual_hash_threshold = 4,
        allowed_app_profiles = { "chromium-research", "libreoffice-headless", "gimp-batch" },
    },

    providers = {
        -- Primary text/VLM planner endpoint (any OpenAI-compatible server).
        primary = {
            kind = "openai_compatible",
            base_url = "https://your-planner-endpoint.example/v1",
            api_key_env = "JIKU_PLANNER_API_KEY",
            thinking = false,
        },
        -- Optional native decision endpoint (requires the google_gemini driver kind).
        decision = {
            kind = "google_gemini",
            base_url = "https://your-decision-endpoint.example/v1beta",
            api_key_env = "JIKU_DECISION_API_KEY",
        },
        -- Optional self-hosted fallback guard; leave `api_key_env` empty when unauthenticated.
        local_vlm = {
            kind = "openai_compatible",
            base_url = "http://127.0.0.1:8080/v1",
            api_key_env = "",
            thinking = false,
        },
    },

    roles = {
        planner = {
            provider = "primary",
            model = "<primary-text-model-id>",
            driver = "turn_sse",
            thinking = false,
            temperature = 0.2,
        },
        decision_router = {
            provider = "decision",
            model = "<decision-model-id>",
            driver = "decisions",
            optional = true,
            fallback_to_primary_text_model = true,
            fast_path_confidence = 0.85,
            max_questions_per_batch = 8,
            require_none_or_other = true,
            require_scope_gate = true,
        },
        fast_guard = {
            provider = "local_vlm",
            model = "<local-vlm-model-id>",
            driver = "turn_sse",
            thinking = false,
            optional = true,
        },
    },

    decision_router = {
        decision_provider = "decision",
        decision_model_id = "<decision-model-id>",
        primary_provider = "primary",
        primary_text_model_id = "<primary-text-model-id>",
        fast_path_confidence_threshold = 0.85,
        max_questions_per_batch = 8,
        compiler = {
            fast_path_confidence = 0.85,
            min_cell_prob = 0.05,
            time_slot_min_confidence = 0.75,
            bernoulli_activation_threshold = 0.80,
            probe_min_prob = 0.15,
            stop_sentinel = "stop_done",
            fallback_token = "NONE_OR_OTHER",
            grid = { cols = 15, rows = 15 },
        },
    },

    memory = {
        backend = "sqlite_fts5",
        db_path = "~/.local/state/jiku/state.db",
        l0_frame_ring_size = 4,
        app_quirk_max_chars = 600,
        stage_new_skills = true,
    },

    notify = {
        desktop_dbus = true,
        app_name = "Jiku",
        notify_on = { "task_completed", "task_failed", "verification_needed", "skill_staged" },
    },

    tts = {
        enabled = false,
        provider = "voicevox",
        endpoint = "http://127.0.0.1:50021",
        speaker_id = 3,
        speak_on = { "task_completed", "verification_needed" },
    },

    avatar = {
        emit_cues = true,
        idle_gaze_follow_vseat = true,
    },
}
