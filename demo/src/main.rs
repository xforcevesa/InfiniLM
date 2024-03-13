use colored::Colorize;
use service::{Device, Service};
use std::{env, io::Write};

const WELCOME_MSG: &str = r#"
###########################################
# 欢迎使用九源推理框架-大模型单机对话demo #
###########################################
"#;
const HELP_MSG: &str = r#"
    /create         新建会话session
    /switch [0-9+]  切换至指定会话
    /help           打印帮助信息

    使用 /exit 或 Ctrl + C 结束程序
"#;

fn main() {
    let path = env::args().nth(1).expect("缺少模型路径参数");
    let infer_service = Service::load_model(path, Device::NvidiaGpu(0));

    println!("{}", WELCOME_MSG);
    println!("{}", HELP_MSG);
    println!("=====================================");

    // 当前会话ID，初始是0
    let mut session_id: usize = 0;
    // 全部会话
    let mut sessions: Vec<service::Session> = Vec::new();
    // 启动会话0
    sessions.push(infer_service.launch());

    loop {
        println!("{}", format!("{}{}:", "会话", &session_id).yellow());
        let mut input = String::new();
        std::io::stdin()
            .read_line(&mut input)
            .expect("Unable to read line.");
        let input = input.trim();

        // 以 / 开头则为用户指令
        if input.starts_with('/') {
            execute_command(input, &mut session_id, &mut sessions, &infer_service);
        } else {
            infer(input, session_id, &mut sessions);
        }
    }
}

fn execute_command(
    command: &str,
    session_id: &mut usize,
    sessions: &mut Vec<service::Session>,
    service: &Service,
) {
    match command {
        "/create" => {
            sessions.push(service.launch());
            *session_id = sessions.len() - 1;
        }
        "/help" => println!("{}", HELP_MSG),
        cmd if cmd.starts_with("/switch") => {
            let target_id: usize = cmd
                .split_whitespace()
                .nth(1)
                .expect("Invalid switch command")
                .parse()
                .expect("Invalid switch command");
            if target_id < sessions.len() {
                *session_id = target_id;
            } else {
                println!("Invalid session ID.")
            }
        }
        "/exit" => std::process::exit(0),
        _ => println!("Unknown Command"),
    }
    println!("=====================================");
}

fn infer(text: &str, session_id: usize, sessions: &mut Vec<service::Session>) {
    println!("{}", "AI:".green());
    sessions[session_id].chat(text, |s| {
        print!("{s}");
        std::io::stdout().flush().unwrap();
    });
    println!("");
}
