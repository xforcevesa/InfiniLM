//! safetensors 文件的加载和访问。

use crate::FileLoadError::{self, Io, Json};
use memmap2::Mmap;
use std::{
    collections::{hash_map, HashMap},
    fs::File,
    io::{Error as IoError, ErrorKind::NotFound},
    mem::size_of_val,
    ops::Deref,
    path::Path,
    pin::Pin,
    sync::Arc,
};

pub use safetensors::{tensor::TensorInfo, Dtype};

/// safetensors 文件的统一结构。
pub struct SafeTensors {
    tensors: HashMap<String, (usize, TensorInfo)>, // name -> (file_index, tensor_info)
    files: Vec<(Mmap, String)>,                    // file_index -> (mmap, format)
}

/// safetensors 文件中的张量映射。
#[derive(Debug)]
pub struct SafeTensor<'a> {
    /// 数据类型。
    pub dtype: Dtype,
    /// 形状。
    pub shape: &'a [usize],
    /// 数据。
    pub data: &'a [u8],
    /// 文件格式。
    pub format: &'a str,
}

/// [SafeTensors] 的张量迭代器。
pub struct Iter<'a> {
    obj: &'a SafeTensors,
    iter: hash_map::Iter<'a, String, (usize, TensorInfo)>,
}

impl SafeTensors {
    /// 自动从路径中加载 safetensors 文件。
    pub fn load_from_dir(path: impl AsRef<Path>) -> Result<Self, FileLoadError> {
        // 先尝试加载单个文件
        let single_file = path.as_ref().join("model.safetensors");
        if single_file.is_file() {
            return Self::single_file(single_file);
        }
        // 再尝试加载索引文件
        let index_file = path.as_ref().join("model.safetensors.index.json");
        if index_file.is_file() {
            return Self::index_file(index_file);
        }
        // 都没有找到
        Err(Io(IoError::new(
            NotFound,
            "No valid safetensors file found",
        )))
    }

    /// 加载单个 `.safetensors` 文件。
    pub fn single_file(path: impl AsRef<Path>) -> Result<Self, FileLoadError> {
        let file = File::open(path).map_err(Io)?;
        let file = unsafe { Mmap::map(&file) }.map_err(Io)?;
        let header = load_header(&file)?;
        Ok(Self {
            tensors: header
                .tensors
                .into_iter()
                .map(|(name, info)| (name, (0, info)))
                .collect(),
            files: vec![(file, header.metadata.format)],
        })
    }

    /// 加载 `.safetensors.index.json` 索引文件。
    pub fn index_file(path: impl AsRef<Path>) -> Result<Self, FileLoadError> {
        use std::collections::hash_map::Entry;
        // 生成目录路径
        let dir = path
            .as_ref()
            .parent()
            .ok_or(IoError::new(NotFound, "Index file has no parent directory"))
            .map_err(Io)?;
        // 加载索引文件
        let index = File::open(&path).map_err(Io)?;
        let index = unsafe { Mmap::map(&index) }.map_err(Io)?;
        let index: SafeTensorsIndex = serde_json::from_slice(&index).map_err(Json)?;
        // 初始化状态
        let mut tensors = HashMap::new();
        let mut file_map = HashMap::new();
        let mut files = Vec::new();
        // 迭代所有张量
        for (name, filename) in index.weight_map {
            match file_map.entry(filename) {
                // 张量在已经加载的文件中
                Entry::Occupied(e) => {
                    let (i, _) = tensors.get(&name).unwrap();
                    assert_eq!(i, e.get());
                }
                // 张量在新文件中
                Entry::Vacant(e) => {
                    // 打开文件
                    let file = File::open(dir.join(e.key())).map_err(Io)?;
                    let file = unsafe { Mmap::map(&file) }.map_err(Io)?;
                    let header = load_header(&file)?;
                    // 迭代文件中的张量
                    let i = files.len();
                    let mut contains = false;
                    for (name_, info) in header.tensors {
                        contains = contains || name == name_;
                        assert!(tensors.insert(name_, (i, info)).is_none());
                    }
                    assert!(contains);
                    // 记录文件映射
                    e.insert(i);
                    files.push((file, header.metadata.format));
                }
            };
        }

        Ok(Self { tensors, files })
    }

    /// 共享自身。
    #[inline]
    pub fn share(self) -> Pin<Arc<Self>> {
        Pin::new(Arc::new(self))
    }

    /// 从共享的 [SafeTensors] 中获取共享的张量。
    pub fn share_tensor(self: &Pin<Arc<Self>>, name: &str) -> Option<SharedTensor> {
        let value = self.tensors.get(name)?;
        let data = self.get_internal(value.0, &value.1).data;
        Some(SharedTensor {
            safetensors: self.clone(),
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
    pub fn get(&self, name: &str) -> Option<SafeTensor> {
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

    fn get_internal<'a>(&'a self, i: usize, info: &'a TensorInfo) -> SafeTensor<'a> {
        let (file, format) = &self.files[i];
        let header_len = unsafe { *file.as_ptr().cast::<u64>() };
        let (begin, end) = info.data_offsets;
        SafeTensor {
            dtype: info.dtype,
            shape: &info.shape,
            data: &file[size_of_val(&header_len)..][header_len as _..][begin..end],
            format,
        }
    }
}

impl<'a> IntoIterator for &'a SafeTensors {
    type Item = (&'a str, SafeTensor<'a>);
    type IntoIter = Iter<'a>;
    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<'a> Iterator for Iter<'a> {
    type Item = (&'a str, SafeTensor<'a>);
    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        self.iter
            .next()
            .map(|(name, (i, info))| (&**name, self.obj.get_internal(*i, info)))
    }
}

/// 共享的张量。
#[derive(Clone)]
pub struct SharedTensor {
    safetensors: Pin<Arc<SafeTensors>>,
    value: &'static (usize, TensorInfo),
    data: &'static [u8],
}

impl Deref for SharedTensor {
    type Target = [u8];
    #[inline]
    fn deref(&self) -> &Self::Target {
        self.data
    }
}

impl SharedTensor {
    /// 数据类型。
    #[inline]
    pub fn dtype(&self) -> Dtype {
        self.value.1.dtype
    }

    /// 形状。
    #[inline]
    pub fn shape(&self) -> &[usize] {
        &self.value.1.shape
    }

    /// 文件格式。
    #[inline]
    pub fn format(&self) -> &str {
        &self.safetensors.files[self.value.0].1
    }

    /// 数据。
    #[inline]
    pub fn data(&self) -> &[u8] {
        self.data
    }
}

#[allow(missing_docs)]
#[derive(serde::Serialize, serde::Deserialize, Debug)]
pub struct SafeTensorsIndex {
    pub metadata: SafeTensorsIndexMetadata,
    pub weight_map: HashMap<String, String>,
}

#[allow(missing_docs)]
#[derive(serde::Serialize, serde::Deserialize, Debug)]
pub struct SafeTensorsIndexMetadata {
    pub total_size: usize,
}

#[allow(missing_docs)]
#[derive(serde::Serialize, serde::Deserialize, Debug)]
pub struct SafeTensorsHeader {
    #[serde(flatten)]
    pub tensors: HashMap<String, TensorInfo>,
    #[serde(rename = "__metadata__")]
    pub metadata: SafeTensorsHeaderMetadata,
}

#[allow(missing_docs)]
#[derive(serde::Serialize, serde::Deserialize, Debug)]
pub struct SafeTensorsHeaderMetadata {
    pub format: String,
}

fn load_header(file: &Mmap) -> Result<SafeTensorsHeader, FileLoadError> {
    let header_len = unsafe { *file.as_ptr().cast::<u64>() };
    let header = &file[size_of_val(&header_len)..][..header_len as _];
    serde_json::from_slice(header).map_err(Json)
}

#[test]
fn test() {
    let Some(model_dir) = crate::test_model::find() else {
        return;
    };
    let safetensors = match SafeTensors::load_from_dir(model_dir) {
        Ok(s) => s,
        Err(FileLoadError::Io(e)) if e.kind() == NotFound => return,
        Err(e) => panic!("{e:?}"),
    };
    println!(
        "found {} tensors in {} files",
        safetensors.tensors_count(),
        safetensors.files_count(),
    );
}
