import numpy as np 
class TaskGenerator():
    def __init__(self,
                 min_bit_arrive,
                 max_bit_arrive,
                 task_arrive_probability,
                 episode_time):
    
    
        self.min_bit_arrive = min_bit_arrive
        self.max_bit_arrive = max_bit_arrive
        self.task_arrive_probability = task_arrive_probability
        self.episode_time = episode_time
    
        self.reset()
    
    def reset(self):
        self.current_time = 0
      
        
    def step(self):
        if self.current_time <= self.episode_time: 
            bitarrive=  np.random.uniform(self.min_bit_arrive, self.max_bit_arrive)
            bitarrive = bitarrive * (np.random.uniform(0, 1)< self.task_arrive_probability)
        else:
            bitarrive = 0
        self.current_time +=1
        return bitarrive
    
t= TaskGenerator(0,1,0.5,100)
x = t.step()
print(x)