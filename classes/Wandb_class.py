# Wandb class

class Wandb_unit:
    def __init__(self):
        self.run = None
    
    def start(self, proj: str, conf: dict):
        if self.run is None:
            self.run = wandb.init(
                project=proj,

                config=conf
            )
            return True
        else:
            print("You must to finish running process before start new one!\nUse fin() method to finish!")
            return False
    
    def fin(self):
        self.run.finish()
        self.run = None
        return True

