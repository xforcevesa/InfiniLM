//! 加载 gguf 文件。

use gguf_rs::{get_gguf_container, GGMLType, Tensor};
use memmap2::Mmap;
use std::{collections::{hash_map, HashMap}, fs::File, mem::size_of_val, ops::Deref, path::Path, pin::Pin, sync::Arc};

use crate::FileLoadError;

/// gguf 文件的统一结构。
pub struct Ggufs {
    tensors: HashMap<String, (usize, Tensor)>, // name -> (file_index, tensor_info)
    files: Vec<(Mmap, String)>,                    // file_index -> (mmap, format)
}

/// gguf 文件中的张量映射。
#[derive(Debug)]
pub struct Gguf<'a> {
    /// 数据类型。
    pub dtype: GGMLType,
    /// 形状。
    pub shape: &'a [u64],
    /// 数据。
    pub data: &'a [u8],
    /// 文件格式。
    pub format: &'a str,
}

/// [Ggufs] 的张量迭代器。
pub struct Iter<'a> {
    obj: &'a Ggufs,
    iter: hash_map::Iter<'a, String, (usize, Tensor)>,
}

/// 共享的张量。
#[derive(Clone)]
pub struct SharedGguf {
    ggufs: Pin<Arc<Ggufs>>,
    value: &'static (usize, Tensor),
    data: &'static [u8],
}

impl Ggufs {
    /// 自动从路径中加载 gguf 文件。
    pub fn load_from_dir(_path: impl AsRef<Path>) -> Result<Self, FileLoadError> {
        Err(FileLoadError::Io(std::io::Error::new(std::io::ErrorKind::InvalidInput, "Not supported yet")))
    }

    /// 从单个 gguf 文件中加载张量。
    pub fn single_file(path: impl AsRef<Path>) -> Result<Self, FileLoadError>  {
        let path = path.as_ref().to_str().unwrap();
        let mut container = match get_gguf_container(path) {
            Ok(container) => container,
            Err(e) => return Err(FileLoadError::Io(std::io::Error::new(std::io::ErrorKind::InvalidData, e))),
        };
        let model = match container.decode() {
            Ok(container) => container,
            Err(e) => return Err(FileLoadError::Io(std::io::Error::new(std::io::ErrorKind::InvalidData, e))),
        };
        let mut tensors = HashMap::new();
        model.tensors().into_iter().for_each(|t|{
            tensors.insert(t.name.clone(), (0, t.clone()));
        });
        let file = File::open(path).unwrap();
        let file = unsafe { Mmap::map(&file) }.unwrap();
        Ok(Self {
            tensors,
            files: vec![(file, String::from("gguf"))],
        })
    }

    /// 从索引文件中加载张量。
    pub fn index_file(_path: impl AsRef<Path>) -> Result<Self, FileLoadError> {
        Err(FileLoadError::Io(std::io::Error::new(std::io::ErrorKind::InvalidInput, "Not supported yet")))
    }

    /// 共享自身。
    #[inline]
    pub fn share(self) -> Pin<Arc<Self>> {
        Pin::new(Arc::new(self))
    }

    /// 从共享的 [SafeTensors] 中获取共享的张量。
    pub fn share_tensor(self: &Pin<Arc<Self>>, name: &str) -> Option<SharedGguf> {
        let value = self.tensors.get(name)?;
        let data = self.get_internal(value.0, &value.1).data;
        Some(SharedGguf {
            ggufs: self.clone(),
            value: unsafe { &*(value as *const _) },
            data: unsafe { &*(data as *const _) },
        })
    }

    /// 检查张量是否存在。
    #[inline]
    pub fn contains(&self, name: &str) -> bool {
        self.tensors.contains_key(name)
    }

    /// 获取张量。
    #[inline]
    pub fn get(&self, name: &str) -> Option<Gguf> {
        self.tensors
            .get(name)
            .map(|(i, info)| self.get_internal(*i, info))
    }

    /// 获取文件数量。
    #[inline]
    pub fn files_count(&self) -> usize {
        self.files.len()
    }

    /// 获取张量数量。
    #[inline]
    pub fn tensors_count(&self) -> usize {
        self.tensors.len()
    }

    /// 获取张量迭代器。
    #[inline]
    pub fn iter(&self) -> Iter {
        Iter {
            obj: self,
            iter: self.tensors.iter(),
        }
    }

    fn get_internal<'a>(&'a self, i: usize, info: &'a Tensor) -> Gguf<'a> {
        let (file, format) = &self.files[i];
        let header_len = unsafe { *file.as_ptr().cast::<u64>() };
        let (begin, end) = (info.offset, info.offset + info.size);

        Gguf {
            dtype: info.kind.try_into().unwrap(),
            shape: info.shape.as_slice(),
            data: &file[size_of_val(&header_len)..][header_len as _..][begin as usize..end as usize],
            format,
        }
    }
}

impl<'a> IntoIterator for &'a Ggufs {
    type Item = (&'a str, Gguf<'a>);
    type IntoIter = Iter<'a>;
    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<'a> Iterator for Iter<'a> {
    type Item = (&'a str, Gguf<'a>);
    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        self.iter
            .next()
            .map(|(name, (i, info))| (&**name, self.obj.get_internal(*i, info)))
    }
}

impl Deref for SharedGguf {
    type Target = [u8];
    #[inline]
    fn deref(&self) -> &Self::Target {
        self.data
    }
}

impl SharedGguf {
    /// 数据类型。
    #[inline]
    pub fn dtype(&self) -> GGMLType {
        self.value.1.kind.try_into().unwrap()
    }

    /// 形状。
    #[inline]
    pub fn shape(&self) -> &[u64] {
        &self.value.1.shape
    }

    /// 文件格式。
    #[inline]
    pub fn format(&self) -> &str {
        &self.ggufs.files[self.value.0].1
    }

    /// 数据。
    #[inline]
    pub fn data(&self) -> &[u8] {
        self.data
    }
}

#[allow(missing_docs)]
#[derive(serde::Serialize, serde::Deserialize, Debug)]
pub struct GgufsIndex {
    pub metadata: GgufsIndexMetadata,
    pub weight_map: HashMap<String, String>,
}

#[allow(missing_docs)]
#[derive(serde::Serialize, serde::Deserialize, Debug)]
pub struct GgufsIndexMetadata {
    pub total_size: usize,
}

#[allow(missing_docs)]
#[derive(serde::Serialize, serde::Deserialize, Debug)]
pub struct GgufsHeader {
    #[serde(flatten)]
    pub tensors: HashMap<String, Tensor>,
    #[serde(rename = "__metadata__", default = "default_metadata")]
    pub metadata: GgufsHeaderMetadata,
}

#[inline]
fn default_metadata() -> GgufsHeaderMetadata {
    GgufsHeaderMetadata {
        format: "pt".into(),
    }
}

#[allow(missing_docs)]
#[derive(serde::Serialize, serde::Deserialize, Debug)]
pub struct GgufsHeaderMetadata {
    pub format: String,
}

fn load_header(file: &Mmap) -> Result<GgufsHeader, FileLoadError> {
    let header_len = unsafe { *file.as_ptr().cast::<u64>() };
    let header = &file[size_of_val(&header_len)..][..header_len as _];
    serde_json::from_slice(header).map_err(FileLoadError::Json)
}

#[test]
fn test() {
    let Some(model_dir) = crate::test_model::find() else {
        return;
    };
    let safetensors = match Ggufs::load_from_dir(model_dir) {
        Ok(s) => s,
        Err(FileLoadError::Io(e)) if e.kind() == std::io::ErrorKind::NotFound => return,
        Err(e) => panic!("{e:?}"),
    };
    println!(
        "found {} tensors in {} files",
        safetensors.tensors_count(),
        safetensors.files_count(),
    );
}

