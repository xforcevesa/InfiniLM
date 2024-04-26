use std::{
    env, fs,
    path::{Path, PathBuf},
    process::Command,
    str::from_utf8,
};

#[derive(Args, Default)]
pub struct DeployArgs {
    #[clap(long, short)]
    target: Option<PathBuf>,
}

impl DeployArgs {
    pub fn deploy(self) {
        let output = Command::new(env!("CARGO"))
            .arg("locate-project")
            .arg("--workspace")
            .arg("--message-format=plain")
            .output()
            .unwrap()
            .stdout;
        let workspace = Path::new(from_utf8(&output).unwrap().trim())
            .parent()
            .unwrap();

        let exe = env::current_exe().unwrap();
        let target = self.target.unwrap_or_else(|| workspace.join("deploy"));
        let target = if target.is_file() {
            fs::remove_file(&target).unwrap();
            target
        } else if target.is_dir() {
            target.join(exe.file_name().unwrap())
        } else if !target.exists() {
            fs::create_dir_all(&target).unwrap();
            target.join(exe.file_name().unwrap())
        } else {
            panic!("Target already exists, but is not a file or directory.");
        };

        println!("Deploy");
        println!("    *- {}", exe.display());
        println!("    -> {}", target.display());
        fs::copy(exe, target).unwrap();
    }
}
