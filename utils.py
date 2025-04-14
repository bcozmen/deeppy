import inspect

import torch.optim as optim
import torch.nn as nn


def plot_scheduler(scheduler, params, step = 1000):
    class NullModule(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(1,1)
    

    model = NullModule()
    optimizer = optim.Adam(model.parameters(), lr = 1)

    def plot_lr(scheduler, optimizer,step=1000):
        lrs = []
        for i in range(step):
            lr = scheduler.get_last_lr()
            scheduler.step()
            lrs.append(lr)

        return lrs
    scheduler = scheduler(optimizer,**params)
    

    return plot_lr(scheduler, optimizer, step=step)

def print_args(self, depth = 0, print_only_this = False):
    if not print_only_this:
        try:
            [dep.print_args(depth = depth + 1) for dep in self.dependencies]
        except:
            pass

    empty = " " * depth * 4

    #If layer generator, print generate
    method = self.__init__
    name = self.__name__
    if name == "LayerGenerator":
        name = "arch"
        method = self.generate

    print(empty + name + "_params = {")
    params = list(inspect.signature(method).parameters.values())[1:]

    if self.__name__ == "Scheduler" or self.__name__ == "Optimizer":
        params = params[1:]

    for s in params:
        s = str(s)
        s = s.split("=")
        s[0] = f"\"{s[0]}\""
        s = ":".join(s)  + ","

        if ("device" in s or "criterion" in s) and depth > 0:
            continue
        print(empty + "    " + s)

    

    print(empty + "}")
    
