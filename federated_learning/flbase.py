class FlBase():
    def __init__(self,averaging_frequency=0,*args, **kwargs):
        self.averaging_frequency = averaging_frequency
        self.counter = 0
        
    def count_epochs(self):
        if self.averaging_frequency:
            self.counter +=1
            self.counter = self.counter % self.averaging_frequency  
            if self.counter == self.averaging_frequency :
                return True
        return False
                
    def average_weights(self,agents,*args,**kwargs):
        pass
    
    
    