"""Thin adapter for the same-run interleaved Mega comparison harness."""
import importlib.util
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
_NAME='_grain_thor_interleaved_mega_local_api'
if _NAME not in sys.modules:
    spec=importlib.util.spec_from_file_location(_NAME,HERE/'local_ctypes_api.py')
    api=importlib.util.module_from_spec(spec)
    sys.modules[_NAME]=api
    spec.loader.exec_module(api)
else:
    api=sys.modules[_NAME]
    if Path(api.__file__).resolve()!=HERE/'local_ctypes_api.py':
        raise RuntimeError('A different experimental Mega API is already loaded')


class Experiment:
    def __init__(self,context,consumer_warps=16,persistent_blocks=0,library=None):
        self.c=context
        self.keywords=dict(scale_x=context.sx,scale_gate_up=context.scale_gate_up,
            scale_down=context.scale_down,token_mask=context.token_mask,
            output_dtype=context.output_dtype,gate_up_layout='interleaved',backend='auto',
            consumer_warps=consumer_warps,persistent_blocks=persistent_blocks,library=library)
        self.metadata=api.build_metadata(library)
        self.metadata.update(scope='Full eager validation, routing/statistics CPU read, per-call allocation, resident interleaved G1/SwiGLU/G256/G2 and separate combine',
            consumer_warps_selected=consumer_warps,persistent_blocks=persistent_blocks,
            bridge=str(HERE/'local_ctypes_api.py'),
            source_files={name:str(HERE/name) for name in self.metadata['source_sha256']},
            builder=str(HERE/'build.py'),resident_hidden=True,global_qh_sh=False,
            reusable_workspace=False,optional_initializer=hasattr(api._load_library(self.metadata['library'])[0],'grain_moe_mega_cluster_initialize'))
    def launch_full(self):
        c=self.c
        return api.moe_ffn_mega(c.qx,c.w_gate_up,c.w_down,c.scores,c.ids,**self.keywords)
    def debug(self):
        c=self.c
        return api.debug_moe_ffn_mega(c.qx,c.w_gate_up,c.w_down,c.scores,c.ids,**self.keywords)


def make_experiment(context,**kwargs):
    return Experiment(context,**kwargs)
