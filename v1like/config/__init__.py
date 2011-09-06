
__all__ = ['get']

from os import path
from pprint import pprint

MY_PATH = path.dirname(path.abspath(__file__))

def get(name, verbose=True):

    config_fname = path.join(MY_PATH, name + '.py')
    config_path = path.abspath(config_fname)

    if verbose: print "Config file:", config_path

    v1like_config = {}
    execfile(config_path, {}, v1like_config)

    model = v1like_config['model']

    if len(model) != 1:
        raise NotImplementedError

    rep, featsel = model[0]
    if verbose:
        print '*'*80
        pprint(rep)

    resize_type = rep['preproc'].get('resize_type', 'input')
    if resize_type == 'output':
        if 'max_edge' not in rep['preproc']:
            raise NotImplementedError
        # add whatever is needed to get output = max_edge
        new_max_edge = rep['preproc']['max_edge']

        preproc_lsum = rep['preproc']['lsum_ksize']
        new_max_edge += preproc_lsum-1

        normin_kshape = rep['normin']['kshape']
        assert normin_kshape[0] == normin_kshape[1]
        new_max_edge += normin_kshape[0]-1

        filter_kshape = rep['filter']['kshape']
        assert filter_kshape[0] == filter_kshape[1]
        new_max_edge += filter_kshape[0]-1

        normout_kshape = rep['normout']['kshape']
        assert normout_kshape[0] == normout_kshape[1]
        new_max_edge += normout_kshape[0]-1

        pool_lsum = rep['pool']['lsum_ksize']
        new_max_edge += pool_lsum-1

        rep['preproc']['max_edge'] = new_max_edge

    return rep, featsel
