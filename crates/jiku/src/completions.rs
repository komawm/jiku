//! Dependency-free shell completion generation for `jiku`.
//!
//! `jiku` parses arguments by hand, so completions come from a declarative
//! table. `table_matches_usage` keeps the table and `print_usage()` in sync.

/// Supported shells.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Shell {
    Bash,
    Zsh,
    Fish,
}

impl Shell {
    /// Parses a shell name as accepted by `jiku completions <shell>`.
    #[must_use]
    pub fn parse(raw: &str) -> Option<Self> {
        match raw.trim().to_ascii_lowercase().as_str() {
            "bash" => Some(Self::Bash),
            "zsh" => Some(Self::Zsh),
            "fish" => Some(Self::Fish),
            _ => None,
        }
    }

    /// Canonical shell name.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Bash => "bash",
            Self::Zsh => "zsh",
            Self::Fish => "fish",
        }
    }
}

/// A single option flag.
pub struct Opt {
    pub long: &'static str,
    pub short: Option<char>,
    pub takes_value: bool,
    pub desc: &'static str,
}

/// A command or subcommand node.
pub struct Cmd {
    pub name: &'static str,
    pub desc: &'static str,
    pub opts: &'static [Opt],
    pub subs: &'static [Cmd],
}

const BIN: &str = "jiku";

const ROOT_OPTS: &[Opt] = &[
    Opt {
        long: "config",
        short: None,
        takes_value: true,
        desc: "Path to config.toml or config.lua",
    },
    Opt {
        long: "policy",
        short: None,
        takes_value: true,
        desc: "Path to the init.lua policy",
    },
    Opt {
        long: "socket",
        short: None,
        takes_value: true,
        desc: "Override the jikud socket path",
    },
    Opt {
        long: "json",
        short: None,
        takes_value: false,
        desc: "Emit machine-readable JSON",
    },
    Opt {
        long: "help",
        short: Some('h'),
        takes_value: false,
        desc: "Print help",
    },
    Opt {
        long: "version",
        short: Some('V'),
        takes_value: false,
        desc: "Print version",
    },
];

const DECIDE_OPTS: &[Opt] = &[
    Opt {
        long: "image",
        short: None,
        takes_value: true,
        desc: "Attach an image for the decision",
    },
    Opt {
        long: "emulate",
        short: None,
        takes_value: false,
        desc: "Force the structured planner",
    },
    Opt {
        long: "question",
        short: None,
        takes_value: true,
        desc: "Repeatable decision question",
    },
    Opt {
        long: "choice",
        short: None,
        takes_value: true,
        desc: "Repeatable choice option",
    },
];

const PLAN_OPTS: &[Opt] = &[
    Opt {
        long: "image",
        short: None,
        takes_value: true,
        desc: "Attach an image for the plan",
    },
    Opt {
        long: "system",
        short: None,
        takes_value: true,
        desc: "System prompt override",
    },
    Opt {
        long: "json-object",
        short: None,
        takes_value: false,
        desc: "Request a JSON object reply",
    },
    Opt {
        long: "stream",
        short: None,
        takes_value: false,
        desc: "Stream the response",
    },
];

const RUN_OPTS: &[Opt] = &[
    Opt {
        long: "workspace",
        short: None,
        takes_value: true,
        desc: "Target Koma workspace id",
    },
    Opt {
        long: "dry-run",
        short: None,
        takes_value: false,
        desc: "Plan without emitting input",
    },
    Opt {
        long: "emulate",
        short: None,
        takes_value: false,
        desc: "Force the structured planner",
    },
];

const SKILLS_SUBS: &[Cmd] = &[
    Cmd {
        name: "list",
        desc: "List staged, approved, and rejected skills",
        opts: &[],
        subs: &[],
    },
    Cmd {
        name: "diff",
        desc: "Show a staged skill diff",
        opts: &[],
        subs: &[],
    },
    Cmd {
        name: "stage",
        desc: "Queue a local skill file for review",
        opts: &[],
        subs: &[],
    },
    Cmd {
        name: "approve",
        desc: "Promote a reviewed skill",
        opts: &[Opt {
            long: "promote-dir",
            short: None,
            takes_value: true,
            desc: "Install directory",
        }],
        subs: &[],
    },
];

const TTS_SUBS: &[Cmd] = &[
    Cmd {
        name: "say",
        desc: "Synthesize and enqueue speech",
        opts: &[
            Opt {
                long: "emotion",
                short: None,
                takes_value: true,
                desc: "Avatar emotion cue",
            },
            Opt {
                long: "no-play",
                short: None,
                takes_value: false,
                desc: "Skip local playback",
            },
            Opt {
                long: "wav-out",
                short: None,
                takes_value: true,
                desc: "Write a WAV file",
            },
        ],
        subs: &[],
    },
    Cmd {
        name: "stop",
        desc: "Barge in and stop playback",
        opts: &[],
        subs: &[],
    },
];

const A11Y_SUBS: &[Cmd] = &[
    Cmd {
        name: "snapshot",
        desc: "Capture the AT-SPI tree as @snap:eN handles",
        opts: &[
            Opt {
                long: "skeleton",
                short: None,
                takes_value: false,
                desc: "Interactable nodes only",
            },
            Opt {
                long: "root",
                short: None,
                takes_value: true,
                desc: "Re-root at a @snap:eN handle",
            },
            Opt {
                long: "app-id",
                short: None,
                takes_value: true,
                desc: "Filter by application id",
            },
            Opt {
                long: "title",
                short: None,
                takes_value: true,
                desc: "Filter by window title",
            },
        ],
        subs: &[],
    },
    Cmd {
        name: "act",
        desc: "Run click, focus, set-value, type, or read-value",
        opts: &[],
        subs: &[],
    },
];

const ROOT_SUBS: &[Cmd] = &[
    Cmd {
        name: "status",
        desc: "Show daemon, model, memory, and Koma status",
        opts: &[],
        subs: &[],
    },
    Cmd {
        name: "validate-config",
        desc: "Validate configuration and policy offline",
        opts: &[],
        subs: &[],
    },
    Cmd {
        name: "migrate-config",
        desc: "Emit an equivalent config.lua",
        opts: &[
            Opt {
                long: "out",
                short: None,
                takes_value: true,
                desc: "Output path",
            },
            Opt {
                long: "force",
                short: None,
                takes_value: false,
                desc: "Overwrite an existing file",
            },
        ],
        subs: &[],
    },
    Cmd {
        name: "decide",
        desc: "Evaluate calibrated Noul/Choice decisions",
        opts: DECIDE_OPTS,
        subs: &[],
    },
    Cmd {
        name: "plan",
        desc: "Run a TurnModelDriver call",
        opts: PLAN_OPTS,
        subs: &[],
    },
    Cmd {
        name: "run",
        desc: "Execute an observe-decide-plan-act task loop",
        opts: RUN_OPTS,
        subs: &[],
    },
    Cmd {
        name: "box",
        desc: "Triage files in a dropzone directory",
        opts: &[],
        subs: &[Cmd {
            name: "classify",
            desc: "Classify the dropzone contents",
            opts: &[],
            subs: &[],
        }],
    },
    Cmd {
        name: "memory",
        desc: "Query persisted episodes",
        opts: &[],
        subs: &[Cmd {
            name: "search",
            desc: "Search episodes with FTS5 BM25",
            opts: &[Opt {
                long: "limit",
                short: None,
                takes_value: true,
                desc: "Maximum results",
            }],
            subs: &[],
        }],
    },
    Cmd {
        name: "skills",
        desc: "Manage staged skills",
        opts: &[],
        subs: SKILLS_SUBS,
    },
    Cmd {
        name: "tts",
        desc: "Speech synthesis and playback",
        opts: &[],
        subs: TTS_SUBS,
    },
    Cmd {
        name: "avatar",
        desc: "Companion stage server",
        opts: &[],
        subs: &[Cmd {
            name: "serve",
            desc: "Serve the HTTP and WebSocket stage",
            opts: &[
                Opt {
                    long: "bind",
                    short: None,
                    takes_value: true,
                    desc: "host:port to bind",
                },
                Opt {
                    long: "vrm",
                    short: None,
                    takes_value: true,
                    desc: "VRM model path",
                },
            ],
            subs: &[],
        }],
    },
    Cmd {
        name: "a11y",
        desc: "Inspect or act on the AT-SPI tree",
        opts: &[],
        subs: A11Y_SUBS,
    },
    Cmd {
        name: "context",
        desc: "Inspect context budgeting",
        opts: &[],
        subs: &[Cmd {
            name: "inspect",
            desc: "Show budget and compaction telemetry",
            opts: &[
                Opt {
                    long: "max-tokens",
                    short: None,
                    takes_value: true,
                    desc: "Budget to simulate",
                },
                Opt {
                    long: "sample-text",
                    short: None,
                    takes_value: true,
                    desc: "Sample text to compact",
                },
            ],
            subs: &[],
        }],
    },
    Cmd {
        name: "bench",
        desc: "Decision and grounding benchmarks",
        opts: &[],
        subs: &[Cmd {
            name: "run",
            desc: "Run the 2x2 ablation benchmark",
            opts: &[
                Opt {
                    long: "offline",
                    short: None,
                    takes_value: false,
                    desc: "Deterministic offline run",
                },
                Opt {
                    long: "live",
                    short: None,
                    takes_value: false,
                    desc: "Run against a live VLM",
                },
                Opt {
                    long: "endpoint",
                    short: None,
                    takes_value: true,
                    desc: "Live endpoint URL",
                },
                Opt {
                    long: "model",
                    short: None,
                    takes_value: true,
                    desc: "Live model id",
                },
                Opt {
                    long: "provider",
                    short: None,
                    takes_value: true,
                    desc: "Live provider id",
                },
                Opt {
                    long: "out-dir",
                    short: None,
                    takes_value: true,
                    desc: "Artifact directory",
                },
                Opt {
                    long: "max-context-tokens",
                    short: None,
                    takes_value: true,
                    desc: "Context ceiling",
                },
            ],
            subs: &[],
        }],
    },
    Cmd {
        name: "mcp",
        desc: "Run the stdio MCP server",
        opts: &[
            Opt {
                long: "stateless",
                short: None,
                takes_value: false,
                desc: "One-shot pipe mode",
            },
            Opt {
                long: "print-tools",
                short: None,
                takes_value: false,
                desc: "Dump the tool schemas",
            },
        ],
        subs: &[],
    },
    Cmd {
        name: "completions",
        desc: "Print a shell completion script",
        opts: &[],
        subs: &[],
    },
];

static ROOT: Cmd = Cmd {
    name: BIN,
    desc: "",
    opts: &[],
    subs: ROOT_SUBS,
};

/// Renders a completion script for `shell`.
#[must_use]
pub fn render(shell: Shell) -> String {
    match shell {
        Shell::Bash => bash(),
        Shell::Zsh => zsh(),
        Shell::Fish => fish(),
    }
}

fn words_for(node: &Cmd) -> Vec<String> {
    let mut words: Vec<String> = Vec::new();
    for opt in node.opts.iter().chain(ROOT_OPTS.iter()) {
        words.push(format!("--{}", opt.long));
        if let Some(short) = opt.short {
            words.push(format!("-{short}"));
        }
    }
    words.extend(node.subs.iter().map(|sub| sub.name.to_owned()));
    words
}

fn nested() -> Vec<(String, &'static Cmd)> {
    fn walk(node: &'static Cmd, prefix: String, acc: &mut Vec<(String, &'static Cmd)>) {
        for sub in node.subs {
            let path = format!("{prefix}{} ", sub.name);
            if !sub.subs.is_empty() || !sub.opts.is_empty() {
                acc.push((path.clone(), sub));
            }
            if !sub.subs.is_empty() {
                walk(sub, path, acc);
            }
        }
    }
    let mut acc = Vec::new();
    walk(&ROOT, String::new(), &mut acc);
    acc
}

fn bash() -> String {
    let mut out = String::new();
    out.push_str("# bash completion for jiku — generated by `jiku completions bash`.\n");
    out.push_str("# Install: jiku completions bash | sudo tee /usr/share/bash-completion/completions/jiku >/dev/null\n");
    out.push_str("_jiku() {\n");
    out.push_str("    local cur=\"${COMP_WORDS[COMP_CWORD]}\" path=\"\" i\n");
    out.push_str("    for ((i = 1; i < COMP_CWORD; i++)); do path+=\"${COMP_WORDS[i]} \"; done\n");
    out.push_str("    case \"$path\" in\n");
    out.push_str(&format!(
        "        \"\") COMPREPLY=($(compgen -W \"{}\" -- \"$cur\")) ;;\n",
        words_for(&ROOT).join(" ")
    ));
    for (path, node) in nested() {
        out.push_str(&format!(
            "        \"{path}\") COMPREPLY=($(compgen -W \"{}\" -- \"$cur\")) ;;\n",
            words_for(node).join(" ")
        ));
    }
    out.push_str("        *) COMPREPLY=() ;;\n");
    out.push_str("    esac\n}\n");
    out.push_str("complete -F _jiku jiku\n");
    out
}

fn zsh() -> String {
    let mut out = String::new();
    out.push_str("#compdef jiku\n");
    out.push_str("# zsh completion for jiku — generated by `jiku completions zsh`.\n");
    out.push_str("# Install: jiku completions zsh | sudo tee /usr/share/zsh/site-functions/_jiku >/dev/null\n");
    out.push_str("_jiku() {\n");
    out.push_str("    local path=\"${(j: :)${(@)words[2,CURRENT-1]}}\"\n");
    out.push_str("    case \"$path\" in\n");
    out.push_str(&format!(
        "        \"\") compadd -- {} ;;\n",
        words_for(&ROOT).join(" ")
    ));
    for (path, node) in nested() {
        out.push_str(&format!(
            "        \"{path}\") compadd -- {} ;;\n",
            words_for(node).join(" ")
        ));
    }
    out.push_str("    esac\n}\n");
    out.push_str("compdef _jiku jiku\n");
    out
}

fn fish_condition(parents: &[&str]) -> String {
    parents
        .iter()
        .map(|parent| format!("__fish_seen_subcommand_from {parent}"))
        .collect::<Vec<_>>()
        .join("; and ")
}

fn fish_flag_line(condition: &str, opt: &Opt) -> String {
    let mut line = format!("complete -c {BIN}");
    if !condition.is_empty() {
        line.push_str(&format!(" -n '{condition}'"));
    }
    line.push_str(&format!(" -l {} -d '{}'", opt.long, opt.desc));
    if let Some(short) = opt.short {
        line.push_str(&format!(" -s {short}"));
    }
    if opt.takes_value {
        line.push_str(" -r");
    }
    line.push('\n');
    line
}

fn fish() -> String {
    let mut out = String::new();
    out.push_str("# fish completion for jiku — generated by `jiku completions fish`.\n");
    out.push_str("# Install: jiku completions fish > ~/.config/fish/completions/jiku.fish\n");
    out.push_str(&format!("complete -c {BIN} -f\n"));
    for sub in ROOT_SUBS {
        out.push_str(&format!(
            "complete -c {BIN} -n __fish_use_subcommand -a {} -d '{}'\n",
            sub.name, sub.desc
        ));
    }
    for opt in ROOT_OPTS {
        out.push_str(&fish_flag_line("", opt));
    }
    for sub in ROOT_SUBS {
        for opt in sub.opts {
            out.push_str(&fish_flag_line(&fish_condition(&[sub.name]), opt));
        }
        let names: Vec<&str> = sub.subs.iter().map(|c| c.name).collect();
        for child in sub.subs {
            out.push_str(&format!(
                "complete -c {BIN} -n '{}; and not __fish_seen_subcommand_from {}' -a {} -d '{}'\n",
                fish_condition(&[sub.name]),
                names.join(" "),
                child.name,
                child.desc
            ));
            for opt in child.opts {
                out.push_str(&fish_flag_line(
                    &fish_condition(&[sub.name, child.name]),
                    opt,
                ));
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_shell_renders_bin_and_subcommands() {
        for shell in [Shell::Bash, Shell::Zsh, Shell::Fish] {
            let script = render(shell);
            assert!(script.contains(BIN), "{shell:?} missing binary name");
            for sub in ROOT_SUBS {
                assert!(script.contains(sub.name), "{shell:?} missing {}", sub.name);
            }
        }
    }

    #[test]
    fn nested_commands_are_completed() {
        for shell in [Shell::Bash, Shell::Zsh, Shell::Fish] {
            let script = render(shell);
            for nested in [
                "classify", "search", "approve", "serve", "snapshot", "inspect",
            ] {
                assert!(script.contains(nested), "{shell:?} missing {nested}");
            }
        }
    }

    #[test]
    fn table_matches_usage() {
        let usage = crate::usage_text();
        let commands = usage
            .split_once("COMMANDS:")
            .map(|(_, rest)| rest)
            .unwrap_or_default();
        for sub in ROOT_SUBS {
            assert!(
                commands.contains(sub.name),
                "completion table lists `{}` but usage does not",
                sub.name
            );
        }
        for line in commands.lines() {
            let name = line.split_whitespace().next().unwrap_or_default();
            if name.is_empty() || name.starts_with('-') || name == "jiku" {
                continue;
            }
            assert!(
                ROOT_SUBS.iter().any(|s| s.name == name),
                "usage documents `{name}` but the completion table does not"
            );
        }
    }
}
