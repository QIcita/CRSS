"""Process-local dispatch policy; uses unmodified official spconv CUDA kernels.

No timing-based algorithm selection, no precision/weight/layout changes.
Frozen for the current fp32 48/96/192-channel, SM89 Utonia context only.
"""
import inspect

def install():
    import spconv.algo as algo
    import spconv.pytorch.ops as ops
    conv = algo.CONV
    if getattr(conv, '_cpsf_fixed', False):
        return
    signature = inspect.signature(conv.tune_and_cache)
    def fixed(*args, **kwargs):
        b = signature.bind(*args, **kwargs); b.apply_defaults(); v = b.arguments
        op = v['op_type'].value
        inp, weight, output = v['inp'], v['weight'], v['output']
        c, k = inp.dim(1), output.dim(1)
        assert tuple(v['arch']) == (8, 9) and c == k and c in (48,96,192)
        choices = conv.get_all_available(inp, weight, output,
            v['layout_i'], v['layout_w'], v['layout_o'], v['arch'],
            v['op_type'], v['mask_width'], v['fp32_accum'], v['use_tf32'],
            v['bias'], v['scale'])
        if op == 0:
            tile = 'm32n64k16' if c == 48 else 'm32n32k32'
            target = f'Simt_f32f32f32f32f32tnt_{tile}m32n32k8A1_200_C301LLL_SK'
        elif op == 1:
            tile = 'm32n32k32' if c == 192 else 'm32n64k16'
            target = f'Simt_f32f32f32f32f32ttt_{tile}m32n32k8A1_200_C311LLL_SK'
        else:
            tile = 'm64n64k8' if c == 192 else 'm32n32k32'
            target = f'Simt_f32f32f32f32f32ntt_{tile}m32n32k8A1_210_C321LLL_SK'
        selected = [d for d in choices if str(d) == target]
        assert len(selected) == 1, (target, [str(d) for d in choices])
        res = algo.BestConvAlgoByProfile(selected[0], v['arch'], splitk=1)
        width = v['mask_width'] if op == 2 else -1
        dynamic = weight.view([k,-1,c]).dim(1) > 32
        key = (inp.dtype, weight.dtype, output.dtype,k,c,*v['arch'],width,dynamic)
        [conv.kc_forward_cache,conv.kc_dgrad_cache,conv.kc_wgrad_cache][op][key] = res
        return res, 0.0
    assert not any([conv.kc_forward_cache,conv.kc_dgrad_cache,conv.kc_wgrad_cache])
    conv.tune_and_cache = fixed
    conv._cpsf_fixed = True
    ops.SPCONV_CPP_GEMM = False
