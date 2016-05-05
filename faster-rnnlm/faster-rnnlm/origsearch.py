import grid_search_util
import sys

_NUM_TASKS = 3

def _SettingsIter(task_id):
    c = 0
    for retry in [3, 5, 10]:
        for context_loss_weight in [0.01, 0.1, 0.2, 1, 2, 3, 10]:
            for hidden_size in [100, 200, 300]:
                for hidden_count in [1, 2, 5]:
                    for hidden_type in ['sigmoid', 'tanh', 'relu', 'gru', 'gru-bias', 'gru-insyn']:
                        for bptt in [5, 8, 10, 15]:
                            c = (c + 1) % _NUM_TASKS
                            if (c != task_id):
                                continue
                            conf = grid_search_util.Configuration(
                                    retry=retry, context_loss_weight=context_loss_weight,
                                    context_size=10, hidden=hidden_size, threads=2,
                                    hidden_count=hidden_count, hidden_type=hidden_type,
                                    bptt=bptt)
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
