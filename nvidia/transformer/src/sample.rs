use causal_lm::SampleArgs;
use common::{f16, utok, Blob};
use common_nv::{
    cuda::{bindings::CUstream, memcpy_d2h, AsRaw, DevByte, DevMem, Stream},
    reslice, reslice_mut,
};
use std::{
    collections::HashMap,
    ffi::{c_int, c_void},
    ptr::{null, null_mut},
    sync::{Mutex, OnceLock},
};

#[allow(unused)]
pub(crate) fn sample_cpu(
    args: impl IntoIterator<Item = (usize, SampleArgs)>,
    logits: &[DevByte],
    voc: usize,
    _stream: &Stream,
) -> Vec<utok> {
    let mut host = Blob::new(logits.len());
    memcpy_d2h(&mut host, logits);

    let logits: &[f16] = reslice(&host);
    args.into_iter()
        .map(|(i, arg)| arg.random(&logits[voc * i..][..voc]))
        .collect()
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Default, Debug)]
#[repr(C)]
struct CubKeyValuePair<K, V> {
    k: K,
    v: V,
}

extern "C" {
    // extern "C" cudaError argmax_half(
    //     void *temp_storage, size_t *temp_storage_bytes,
    //     half const *input, int num_items,
    //     cub::KeyValuePair<int, half> *output,
    //     cudaStream_t stream)
    fn argmax_half(
        temp_storage: *mut c_void,
        temp_storage_bytes: *mut usize,
        input: *const f16,
        num_items: c_int,
        output: *mut CubKeyValuePair<c_int, f16>,
        stream: CUstream,
    ) -> c_int;

    // extern "C" cudaError radix_sort_half(
    //     void *temp_storage, size_t *temp_storage_bytes,
    //     half const *key_in, half *key_out,
    //     unsigned int const *value_in, unsigned int *value_out,
    //     int num_items,
    //     cudaStream_t stream)
    fn radix_sort_half(
        temp_storage: *mut c_void,
        temp_storage_bytes: *mut usize,
        key_in: *const f16,
        key_out: *mut f16,
        value_in: *const u32,
        value_out: *mut u32,
        num_items: c_int,
        stream: CUstream,
    ) -> c_int;

    // extern "C" cudaError inclusive_sum_half(
    //     void *temp_storage, size_t *temp_storage_bytes,
    //     half *data, int num_items,
    //     cudaStream_t stream)
    fn inclusive_sum_half(
        temp_storage: *mut c_void,
        temp_storage_bytes: *mut usize,
        data: *mut f16,
        num_items: c_int,
        stream: CUstream,
    ) -> c_int;

    // extern "C" cudaError partial_softmax_half(
    //     half *data,
    //     float temperature,
    //     unsigned int topk,
    //     cudaStream_t stream)
    fn partial_softmax_half(
        data: *mut f16,
        temperature: f32,
        topk: c_int,
        stream: CUstream,
    ) -> c_int;

    // extern "C" cudaError random_sample_half(
    //     half const *data,
    //     unsigned int const *indices,
    //     unsigned int *index,
    //     float probability,
    //     int topk,
    //     cudaStream_t stream)
    fn random_sample_half(
        data: *const f16,
        indices: *const u32,
        index: *mut u32,
        probability: f32,
        topk: c_int,
        stream: CUstream,
    ) -> c_int;
}

fn prealloc_argmax<'ctx>(stream: &Stream<'ctx>, len: usize) -> DevMem<'ctx> {
    static MAP: OnceLock<Mutex<HashMap<usize, usize>>> = OnceLock::new();
    let len = *MAP
        .get_or_init(|| Default::default())
        .lock()
        .unwrap()
        .entry(len)
        .or_insert_with(|| {
            let mut temp_storage_bytes = 0;
            assert_eq!(0, unsafe {
                argmax_half(
                    null_mut(),
                    &mut temp_storage_bytes,
                    null(),
                    len as _,
                    null_mut(),
                    stream.as_raw(),
                )
            });
            temp_storage_bytes
        });
    stream.malloc::<u8>(len)
}

fn prealloc_radix_sort<'ctx>(stream: &Stream<'ctx>, len: usize) -> DevMem<'ctx> {
    static MAP: OnceLock<Mutex<HashMap<usize, usize>>> = OnceLock::new();
    let len = *MAP
        .get_or_init(|| Default::default())
        .lock()
        .unwrap()
        .entry(len)
        .or_insert_with(|| {
            let mut temp_storage_bytes = 0;
            assert_eq!(0, unsafe {
                radix_sort_half(
                    null_mut(),
                    &mut temp_storage_bytes,
                    null(),
                    null_mut(),
                    null(),
                    null_mut(),
                    len as _,
                    stream.as_raw(),
                )
            });
            temp_storage_bytes
        });
    stream.malloc::<u8>(len)
}

fn prealloc_inclusive_sum<'ctx>(stream: &Stream<'ctx>, len: usize) -> DevMem<'ctx> {
    static MAP: OnceLock<Mutex<HashMap<usize, usize>>> = OnceLock::new();
    let len = *MAP
        .get_or_init(|| Default::default())
        .lock()
        .unwrap()
        .entry(len)
        .or_insert_with(|| {
            let mut temp_storage_bytes = 0;
            assert_eq!(0, unsafe {
                inclusive_sum_half(
                    null_mut(),
                    &mut temp_storage_bytes,
                    null_mut(),
                    len as _,
                    stream.as_raw(),
                )
            });
            temp_storage_bytes
        });
    stream.malloc::<u8>(len)
}

#[allow(unused)]
pub(crate) fn sample_nv(
    args: impl IntoIterator<Item = (usize, SampleArgs)>,
    logits: &[DevByte],
    voc: usize,
    stream: &Stream,
) -> Vec<utok> {
    let mut temp_argmax = prealloc_argmax(&stream, voc);
    let mut argmax_host = CubKeyValuePair::<c_int, f16>::default();
    let mut argmax_out = stream.malloc::<CubKeyValuePair<c_int, f16>>(1);

    let mut temp_sort = prealloc_radix_sort(&stream, voc);
    let mut sort_out = stream.malloc::<f16>(voc);
    let mut indices_host = stream.ctx().malloc_host::<u32>(voc);
    reslice_mut::<u8, u32>(&mut indices_host)
        .iter_mut()
        .enumerate()
        .for_each(|(i, idx)| *idx = i as u32);
    let mut indices_in = stream.from_host(&indices_host);
    let mut indices_out = stream.malloc::<u32>(voc);

    let mut temp_sum = prealloc_inclusive_sum(&stream, voc);

    let logits = logits.as_ptr().cast::<f16>();
    let ans = args
        .into_iter()
        .map(|(i, args)| {
            let logits = unsafe { logits.add(i * voc) };

            if args.is_argmax() {
                assert_eq!(0, unsafe {
                    argmax_half(
                        temp_argmax.as_mut_ptr().cast(),
                        &mut temp_argmax.len(),
                        logits,
                        voc as _,
                        argmax_out.as_mut_ptr().cast(),
                        stream.as_raw(),
                    )
                });
                memcpy_d2h(std::slice::from_mut(&mut argmax_host), &argmax_out);
                argmax_host.k as utok
            } else {
                let topk = args.top_k.min(voc) as c_int;
                assert_eq!(0, unsafe {
                    radix_sort_half(
                        temp_sort.as_mut_ptr().cast(),
                        &mut temp_sort.len(),
                        logits,
                        sort_out.as_mut_ptr().cast(),
                        indices_in.as_ptr().cast(),
                        indices_out.as_mut_ptr().cast(),
                        voc as _,
                        stream.as_raw(),
                    )
                });
                assert_eq!(0, unsafe {
                    partial_softmax_half(
                        sort_out.as_mut_ptr().cast(),
                        args.temperature,
                        topk,
                        stream.as_raw(),
                    )
                });
                assert_eq!(0, unsafe {
                    inclusive_sum_half(
                        temp_sum.as_mut_ptr().cast(),
                        &mut temp_sum.len(),
                        sort_out.as_mut_ptr().cast(),
                        topk,
                        stream.as_raw(),
                    )
                });
                let mut index = 0;
                assert_eq!(0, unsafe {
                    random_sample_half(
                        sort_out.as_ptr().cast(),
                        indices_out.as_ptr().cast(),
                        &mut index,
                        rand::random::<f32>() * args.top_p,
                        topk,
                        stream.as_raw(),
                    )
                });
                index as utok
            }
        })
        .collect();

    temp_argmax.drop_on(&stream);
    argmax_out.drop_on(&stream);

    temp_sort.drop_on(&stream);
    sort_out.drop_on(&stream);
    indices_in.drop_on(&stream);
    indices_out.drop_on(&stream);

    temp_sum.drop_on(&stream);

    ans
}
