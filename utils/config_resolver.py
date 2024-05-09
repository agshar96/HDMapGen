'''
This file resolves various config flags. As they work with a specific setting.
'''
def large_scale_error_check(config):
    if not config.dataset.has_node_feat:
        print("In current setting large scale graph requires node embedding")
        config.dataset.has_node_feat = True
    if not config.dataset.has_sub_nodes:
        print("In current setting large scale graph gen requires subnodes to be enabled")
        config.dataset.has_sub_nodes = True
    if config.test.is_vis:
        print("For large graph gen we turn normal visulization off")
        config.test.is_vis = False
    
def resolve_config(config):
    if not hasattr(config.dataset, "has_node_feat"):
        config.dataset.has_node_feat = False
    if not hasattr(config.dataset, "has_sub_nodes"):
        config.dataset.has_sub_nodes = False
    if not hasattr(config.dataset, "has_stop_node"):
        config.dataset.has_stop_node = False
    if not hasattr(config.test, "large_scale_gen"):
        config.test.large_scale_gen = False

    #### WARNING
    if config.dataset.has_stop_node:
        if not config.dataset.has_sub_nodes:
            print("WARNING-- Stop nodes have been tested mainly with subnodes enabled!")
    
    if config.test.large_scale_gen:
        large_scale_error_check(config)
        
    if config.dataset.has_sub_nodes:
        config.dataset.has_node_feat = True

    if config.dataset.has_stop_node:
        config.model.max_num_nodes += 1

    return config