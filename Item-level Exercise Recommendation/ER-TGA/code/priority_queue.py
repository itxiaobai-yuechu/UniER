import queue
from scipy.optimize import linear_sum_assignment

class Alliance(object):
    def __init__(self, Id, u, v, Historical_Marriage_Iterations, Distance_Last_Marriage_Iterations):
        self.Id = Id
        self.u = u; self.v = v
        self.Historical_Marriage_Iterations = Historical_Marriage_Iterations
        self.Distance_Last_Marriage_Iterations = Distance_Last_Marriage_Iterations

    def __lt__(self, other):
        if self.Historical_Marriage_Iterations == other.Historical_Marriage_Iterations:
            return self.Distance_Last_Marriage_Iterations > other.Distance_Last_Marriage_Iterations
        return self.Historical_Marriage_Iterations < other.Historical_Marriage_Iterations

n = 5
Edge_Weight = [[-1, -1, 0, 0] for i in range((int)(n*(n-1)/2))]
cnt = 0
vis = [False for i in range(n+1)]
for i in range(1, n + 1):
    for j in range(i + 1, n + 1):
        Edge_Weight[cnt] = [i, j, i, j]
        cnt += 1
que = queue.PriorityQueue()
for num in range((int)(n*(n-1)/2)):
    [x, y, u, v] = Edge_Weight[num]
    que.put(Alliance(Id = num, u = u, v = v, Historical_Marriage_Iterations=x, Distance_Last_Marriage_Iterations=y))

while que.empty() == False:
    Top = que.get()
    u = Top.u; v = Top.v; num = Top.Id
    if vis[u] == True or vis[v] == True:
        continue
    print("(%d, %d)"%(u, v))
    vis[u] = vis[v] = True
    Edge_Weight[num][2] += 1
    Edge_Weight[num][3] = 0
