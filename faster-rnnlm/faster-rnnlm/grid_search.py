import grid_search_util
import os
import sys

_NUM_TASKS = 11


def _SettingsIter(task_id):
    c = 0
    vocab_settings = {
            10: '../../pbamotra_data/data_4',
            20: '../../pbamotra_data/data_6',
            30: '../../pbamotra_data/data_10',
            40: '../../pbamotra_data/data_11'}

    pos_size = 45
    for hidden_type in ['sigmoid', 'gru-insyn']:
        for retry in [20]:
            for hidden_size in [128, 256]:
                for context_loss_weight in [0.01, 0.001, 0.5]:
                    for hidden_count in [1, 3, 5]:
                        for nce in [0, 30]:
                            for alpha in [0.1, 0.2]:
                                for context_loss_type in [1, 2, 3]:
                                    for lda_size, fpath in vocab_settings.items():
                                        beta_file = os.path.join(fpath, 'lda_betas.csv')
                                        dict_file = os.path.join(fpath, 'dictionary.csv')
                                        pos_file = os.path.join(fpath, 'pt_mat.csv')

                                        c = (c + 1) % _NUM_TASKS
                                        if c != task_id:
                                            continue
                                        conf = grid_search_util.Configuration(
                                                retry=retry,
                                                context_loss_weight=context_loss_weight,
                                                context_size=lda_size, # the code add pos_size internally.
                                                hidden=hidden_size + lda_size + pos_size,
                                                nce=nce,
                                                alpha=alpha,
                                                nce_accurate_test=1,
                                                context_loss_type=context_loss_type,
                                                beta_filepath=beta_file,
                                                threads=3,
                                                pos_context_size=0, # DISABLE POS.
                                                dict_filepath=dict_file,
                                                pos_filepath=pos_file)
                                        yield conf


def _RunTaskId(task_id):
    c = 0
    for conf in _SettingsIter(task_id):
        c += 1
        print 'Task(%d): Running %d, %s' % (task_id, c, conf.GetModelName())
        conf.Run()

def Main():
    grid_search_util.Compile()
    if (len(sys.argv) != 2):
        print 'Usage: %s <task-id>' % sys.argv[0]
        sys.exit(1)
    task_id = int(sys.argv[1])
    _RunTaskId(task_id)

if __name__ == '__main__':
    Main()
