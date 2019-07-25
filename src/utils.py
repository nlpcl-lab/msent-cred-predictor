import os
import glob
import pprint
import argparse

pp = pprint.PrettyPrinter(indent=2)


def pretty_print(to_outs):
    '''
    Get list of name-value pairs, and print them in more readable form
    '''
    print('\n****************')
    for name, value in to_outs:
        print('%s: %s' % (name, pp.pformat(value)))
    print('****************\n')


def str2bool(v):
    if v.lower() in ('true', 't', 'yes', 'y', '1'):
        return True
    elif v.lower() in ('false', 'f', 'no', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def get_docnames(path):
    dlist = filter(lambda x: x.endswith('.tsv'), os.listdir(path))
    return list(map(lambda x: os.path.join(path, x), dlist))
