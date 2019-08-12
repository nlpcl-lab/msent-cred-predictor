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


def truncate_seq_pair(tok_a, tok_b, max_length):
    '''Truncates a sequence pair in place to the maximum length. (From pytorch-transformers)'''

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    la, lb = len(tok_a), len(tok_b)

    while True:
        total_length = la + lb
        if total_length <= max_length:
            break

        if la > lb:
            la -= 1
        else:
            lb -= 1

    return (la, lb)
