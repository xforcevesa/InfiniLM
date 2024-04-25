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
        let target = self
            .target
            .unwrap_or_else(|| workspace.join("deploy"))
            .join(exe.file_name().unwrap());
        println!("Deploy");
        println!("    from: {}", exe.display());
        println!("    to:   {}", target.display());

        fs::create_dir_all(target.parent().unwrap()).unwrap();
        fs::remove_file(&target).unwrap();
        fs::copy(exe, target).unwrap();
    }
}
