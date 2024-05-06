//! 从环境变量中获得测试模型路径的函数。

use std::{
    env::var_os,
    fs::canonicalize,
    path::{Path, PathBuf},
    process::Command,
    str::from_utf8,
};

/// 从环境变量中获得测试模型路径。
pub fn find() -> Option<PathBuf> {
    let model = var_os("TEST_MODEL")?;
    if let Ok(path) = canonicalize(&model) {
        return Some(path);
    }

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
    if let Ok(path) = canonicalize(workspace.join(model)) {
        return Some(path);
    }

    None
}

#[test]
fn test_find() {
    println!("{:?}", find());
}
