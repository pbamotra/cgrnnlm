import grid_search_util
import sys

_NUM_TASKS = 4

def _SettingsIter(task_id):
    c = 0
    context_size = 10
    for hidden_type in ['sigmoid', 'gru-insyn']:
        for retry in [10, 20]:
            for hidden_size in [128 + context_size, 256 + context_size]:
                for context_loss_weight in [0.01, 0.001]:
                    for hidden_count in [1]:
                        for bptt in [5, 10]:
                            for nce in [20, 40, 50]:
                                for alpha in [0.1, 0.2, 0.3]:
                                    for normalize in [0, 1]:
                                        for l1_loss in [0, 1]:
                                            c = (c + 1) % _NUM_TASKS
                                            if (c != task_id):
                                                continue
                                            conf = grid_search_util.Configuration(
                                                    retry=retry, context_loss_weight=context_loss_weight,
                                                    context_size=context_size, hidden=hidden_size, threads=2,
                                                    hidden_count=hidden_count, hidden_type=hidden_type,
                                                    bptt=bptt, nce=nce, alpha=alpha, nce_accurate_test=1,
                                                    l1_loss=l1_loss,normalize=normalize)
#                                            l1_loss=l1_loss,normalize=normalize)
                                            yield conf

def _RunTaskId(task_id):
    c = 0
    for conf in _SettingsIter(task_id):
        c += 1
        print 'Task(%d): Running %d, %s' % (task_id, c, conf.GetModelName())
        conf.Run()

def Main():
    #grid_search_util.Compile()
    if (len(sys.argv) != 2):
        print 'Usage: %s <task-id>' % sys.argv[0]
        sys.exit(1)
    task_id = int(sys.argv[1])
    _RunTaskId(task_id)

if __name__ == '__main__':
    Main()
