-- ~/.config/jiku/init.lua — Sandboxed Phodopus Lua 5.4 policy & skill hooks for jikud
-- Runs with zero `io` / `os` / `debug` libraries under deterministic InstructionBudget fuel.

-- 1. Shape L1 per-app UI quirks before an episode step is sent to the planner/actor
jiku.memory.on_prefetch(function(ctx)
    if ctx.active_app_id == "org.PWMT.zathura" then
        ctx:append_note("Prefer keyboard navigation (`j`, `k`, `J`, `K`, `/`) over scrollbars.")
    elseif ctx.active_app_id == "org.blender.Blender" then
        ctx:append_note("Wait 80ms settle after modal operator popups before clicking confirm.")
    end
end)

-- 2. Customize completion notification & optional TTS summary text
jiku.on("episode.completed", function(ep)
    jiku.notify({
        summary = "Jiku 任务已完成 (" .. ep.workspace .. ")",
        body = ep.summary,
        urgency = "normal",
        actions = {
            { id = "peek_ws", label = "查看工作区" },
        },
    })

    if ep.duration_ms > 5000 then
        jiku.tts.say("后台任务已经处理完成啦。")
    end
end)

-- 3. Immediately pause & alert the user if Koma reports a Protected Surface / 2FA prompt
jiku.on("security.verification_needed", function(ev)
    jiku.avatar.set_emotion("alert")
    jiku.notify({
        summary = "需要人工确认 (" .. ev.workspace .. ")",
        body = "检测到受保护鉴权窗口或验证码，请按 Mod + A 查看并按 Mod + G 接管。",
        urgency = "critical",
    })
end)
