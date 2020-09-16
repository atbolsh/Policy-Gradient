from PolicyGradientAgent import *
from shortCorridor import *

a = MCAgent()

a.intensities['S'] = np.array([np.log(0.1), np.log(0.9)]) # Need a bad start

episodes = 1000
a.alpha = 0.1 # This value gives good results; almost optimal!
lengths = []
for i in range(episodes):
    e = exampleEnv()
#    a.alpha = 2**(-13)
    lengths.append(a.episode(e))
    print(lengths[-1])

print(a.probs('S'))
print(sum(lengths[-100:])/100.0)

