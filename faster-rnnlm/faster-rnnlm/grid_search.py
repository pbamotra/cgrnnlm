import grid_search_util

def Main():
    grid_search_util.Compile()
    conf = grid_search_util.Configuration(
            retry=3, context_loss_weight=1, context_size=10, hidden=50)
    conf.Run()

if __name__ == '__main__':
    Main()
