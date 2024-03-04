pub(super) trait Channel {
    fn receive(&mut self) -> Result<Query, ReceiveError>;
    fn send(&mut self, response: Response) -> Result<(), SendError>;
}

pub(super) fn channel(arg: Option<String>) -> Box<dyn Channel> {
    arg.as_ref()
        .filter(|a| a.as_str() != "stdio")
        .and_then(|_| todo!())
        .unwrap_or_else(|| {});
    todo!()
}

#[derive(Debug)]
pub(super) enum ReceiveError {
    NoQuery,
    Io(std::io::Error),
    Json(serde_json::Error),
}

#[derive(Debug)]
pub(super) enum SendError {
    Io(std::io::Error),
    Json(serde_json::Error),
}

pub(super) struct StdioChannel;

impl Channel for StdioChannel {
    fn receive(&mut self) -> Result<Query, ReceiveError> {
        let mut buf = String::new();
        std::io::stdin()
            .read_line(&mut buf)
            .map_err(ReceiveError::Io)?;
        serde_json::from_str(&buf).map_err(ReceiveError::Json)
    }

    fn send(&mut self, response: Response) -> Result<(), SendError> {
        let buf = serde_json::to_string(&response).map_err(SendError::Json)?;
        println!("{buf}");
        Ok(())
    }
}

#[derive(serde::Serialize, serde::Deserialize, Debug)]
pub(super) struct Query {
    pub id: usize,
    pub prompt: String,
}

#[derive(serde::Serialize, serde::Deserialize, Debug)]
pub(super) struct Response {
    pub id: usize,
    pub prompt: String,
}
