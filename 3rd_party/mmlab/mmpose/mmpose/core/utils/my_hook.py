from mmcv.runner import HOOKS, Hook


@HOOKS.register_module()
class MyHook(Hook):

    def __init__(self):
        print("hook!!!")

    def before_run(self, runner):
        pass

    def after_run(self, runner):
        pass

    def before_epoch(self, runner):
        pass

    def after_epoch(self, runner):
        pass

    def before_iter(self, runner):
        pass

    def after_iter(self, runner):
        # print(dir(runner))
        pass

"""
dir runner
['__abstractmethods__', '__class__', '__delattr__', '__dict__', '__dir__', '__doc__', '__eq__', '__format__', '__ge__',
 '__getattribute__', '__gt__', '__hash__', '__init__', '__init_subclass__', '__le__', '__lt__', '__module__', '__ne__',
  '__new__', '__reduce__', '__reduce_ex__', '__repr__', '__setattr__', '__sizeof__', '__str__', '__subclasshook__',
   '__weakref__', '_abc_impl', '_epoch', '_hooks', '_inner_iter', '_iter', '_max_epochs', '_max_iters', '_model_name',
    '_rank', '_world_size', 'batch_processor', 'call_hook', 'current_lr', 'current_momentum', 'data_loader', 'epoch', 
    'get_hook_info', 'hooks', 'inner_iter', 'iter', 'load_checkpoint', 'log_buffer', 'logger', 'max_epochs', 
    'max_iters', 'meta', 'mode', 'model', 'model_name', 'optimizer', 'outputs', 'rank', 'register_checkpoint_hook', 
    'register_custom_hooks', 'register_hook', 'register_hook_from_cfg', 'register_logger_hooks', 'register_lr_hook', 
    'register_momentum_hook', 'register_optimizer_hook', 'register_profiler_hook', 'register_timer_hook', 
    'register_training_hooks', 'resume', 'run', 'run_iter', 'save_checkpoint', 'timestamp', 'train', 'val', 'work_dir',
     'world_size']



dir model
['T_destination', '__annotations__', '__call__', '__class__', '__delattr__', '__dict__', '__dir__', '__doc__', '__eq__', 
'__format__', '__ge__', '__getattr__', '__getattribute__', '__gt__', '__hash__', '__init__', '__init_subclass__', 
'__le__', '__lt__', '__module__', '__ne__', '__new__', '__reduce__', '__reduce_ex__', '__repr__', '__setattr__', 
'__setstate__', '__sizeof__', '__str__', '__subclasshook__', '__weakref__', '_apply', '_backward_hooks', '_buffers', 
'_call_impl', '_forward_hooks', '_forward_pre_hooks', '_get_backward_hooks', '_get_name', '_is_full_backward_hook', 
'_load_from_state_dict', '_load_state_dict_pre_hooks', '_maybe_warn_non_full_backward_hook', '_modules', 
'_named_members', '_non_persistent_buffers_set', '_parameters', '_register_load_state_dict_pre_hook', 
'_register_state_dict_hook', '_replicate_for_data_parallel', '_save_to_state_dict', '_slow_forward', 
'_state_dict_hooks', '_version', 'add_module', 'apply', 'bfloat16', 'buffers', 'children', 'cpu', 'cuda', 'device_ids', 
'dim', 'double', 'dump_patches', 'eval', 'extra_repr', 'float', 'forward', 'gather', 'get_buffer', 'get_parameter', 
'get_submodule', 'half', 'load_state_dict', 'module', 'modules', 'named_buffers', 'named_children', 'named_modules', 
'named_parameters', 'output_device', 'parallel_apply', 'parameters', 'register_backward_hook', 'register_buffer', 
'register_forward_hook', 'register_forward_pre_hook', 'register_full_backward_hook', 'register_parameter', 'replicate', 
'requires_grad_', 'scatter', 'share_memory', 'src_device_obj', 'state_dict', 'to', 'to_empty', 'train', 'train_step', 
'training', 'type', 'val_step', 'xpu', 'zero_grad']


"""