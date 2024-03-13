use crate::{init_logger, service};
use ::service::{Service, Session};
use colored::Colorize;
use std::{collections::HashMap, io::Write};

#[derive(Args, Default)]
pub(crate) struct ChatArgs {
    /// Model directory.
    #[clap(short, long)]
    model: String,
    /// Log level, may be "off", "trace", "debug", "info" or "error".
    #[clap(long)]
    log: Option<String>,

    /// Use Nvidia GPU.
    #[clap(long)]
    nvidia: bool,
}

impl ChatArgs {
    pub fn invoke(self) {
        init_logger(self.log);
        let service = service(&self.model, self.nvidia);
        let mut session = service.launch();
        let mut sessions = HashMap::new();

        println!("{}", WELCOME_MSG);
        println!("{}", HELP_MSG);
        println!("=====================================");
        loop {
            println!("{}", format!("会话 {}:", session.id()).yellow());
            let mut input = String::new();
            std::io::stdin()
                .read_line(&mut input)
                .expect("Unable to read line.");

            // 以 / 开头则为用户指令
            if input.trim_start().starts_with('/') {
                execute_command(&input, &mut session, &mut sessions, &service);
            } else {
                infer(&input, &mut session);
            }
        }
    }
}

const WELCOME_MSG: &str = r#"
###########################################
# 欢迎使用九源推理框架-大模型单机对话demo #
###########################################
"#;
const HELP_MSG: &str = r#"
    /create        新建会话session
    /switch [0-9+] 切换至指定会话
    /drop [0-9+]   丢弃指定会话
    /help          打印帮助信息

    使用 /exit 或 Ctrl + C 结束程序
"#;

fn execute_command(
    command: &str,
    session: &mut Session,
    sessions: &mut HashMap<usize, Session>,
    service: &Service,
) {
    match command.split_whitespace().collect::<Vec<_>>().as_slice() {
        ["/create"] => {
            let old = std::mem::replace(session, service.launch());
            sessions.insert(old.id(), old);
        }
        ["/switch", n] => match n.parse() {
            Ok(target_id) => {
                if target_id == session.id() {
                    println!("Already in session {}", target_id);
                } else if let Some(target) = sessions.remove(&target_id) {
                    let old = std::mem::replace(session, target);
                    sessions.insert(old.id(), old);
                } else {
                    println!("Invalid session ID.");
                }
            }
            Err(_) => println!("Invalid drop command"),
        },
        ["/drop", n] => match n.parse() {
            Ok(target_id) => {
                if target_id == session.id() {
                    if let Some((&id, _)) = sessions.iter().next() {
                        let _ = std::mem::replace(session, sessions.remove(&id).unwrap());
                    } else {
                        *session = service.launch();
                    }
                    println!("Session {target_id} is dropped.")
                } else if sessions.remove(&target_id).is_some() {
                    println!("Session {target_id} is dropped.");
                } else {
                    println!("Invalid session ID.");
                }
            }
            Err(_) => println!("Invalid drop command"),
        },
        ["/help"] => println!("{}", HELP_MSG),
        ["/exit"] => std::process::exit(0),
        _ => println!("Unknown Command"),
    }
    println!("=====================================");
}

fn infer(text: &str, session: &mut Session) {
    println!("{}", "AI:".green());
    session.chat(text, |s| match s {
        "\\n" => println!(),
        _ => {
            print!("{s}");
            std::io::stdout().flush().unwrap();
        }
    });
    println!();
}
