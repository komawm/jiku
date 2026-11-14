//! Pure-data command-surface descriptors for the Jiku controller client (`jiku`).
//!
//! This library remains a non-runnable incubation scaffold prior to `v0.1.0`:
//! no CLI binary, TUI loop, MCP server runtime, or socket transport is built
//! yet.

/// Planned command surfaces exposed by the `jiku` client controller over the
/// `jikud` orchestration daemon.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum JikuSurfaceCommand {
    /// Attach the interactive terminal UI (`jiku` default mode).
    AttachTui,
    /// Submit a one-shot task instruction (`jiku run "<task>"`).
    RunTask,
    /// Query current daemon, model driver, and Koma workspace status (`jiku status`).
    Status,
    /// Start or attach to a full-duplex `LiveSessionDriver` co-pilot session (`jiku live`).
    LiveCompanion,
    /// Run multimodal `DecisionsDriver` triage on `~/JikuBox` (`jiku box classify`).
    JikuBoxClassify,
    /// List active and staged Lua / SOP skills (`jiku skills list`).
    SkillsList,
    /// Inspect diff of a staged skill awaiting review (`jiku skills diff <id>`).
    SkillsDiff,
    /// Approve and promote a staged skill (`jiku skills approve <id>`).
    SkillsApprove,
    /// Run the Model Context Protocol bridge (`jiku mcp`, optionally `--stateless`).
    McpServer {
        /// Whether the MCP server runs in stateless request-per-call mode.
        stateless: bool,
    },
}

impl JikuSurfaceCommand {
    /// Returns `true` when the command requires human-in-the-loop approval or
    /// interactive terminal attachment.
    #[must_use]
    pub const fn is_interactive_or_approval_gate(self) -> bool {
        matches!(
            self,
            Self::AttachTui | Self::LiveCompanion | Self::SkillsApprove
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn surface_command_identifies_interactive_and_stateless_mcp_modes() {
        assert!(JikuSurfaceCommand::AttachTui.is_interactive_or_approval_gate());
        assert!(JikuSurfaceCommand::LiveCompanion.is_interactive_or_approval_gate());
        assert!(JikuSurfaceCommand::SkillsApprove.is_interactive_or_approval_gate());
        assert!(!JikuSurfaceCommand::JikuBoxClassify.is_interactive_or_approval_gate());
        assert!(
            !JikuSurfaceCommand::McpServer { stateless: true }.is_interactive_or_approval_gate()
        );
    }
}
