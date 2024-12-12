#Current architecture

class MLP(nn.Module):
    def __init__(self, input_size):
        super(MLP, self).__init__()
        hidden_size1 = int((2/3) * input_size + 1)
        hidden_size2 = hidden_size1 // 2
        hidden_size3 = hidden_size2 // 2

        self.network = nn.Sequential(
            nn.LazyLinear(hidden_size1),
            nn.BatchNorm1d(hidden_size1),
            nn.Sigmoid(),
            nn.LazyLinear(hidden_size2),
            nn.BatchNorm1d(hidden_size2),
            nn.Sigmoid(),
            nn.LazyLinear(hidden_size3),
            nn.BatchNorm1d(hidden_size3),
            nn.Sigmoid(),
            nn.Linear(hidden_size3, 3)  # 3 classes: A, B, C
        )

    def forward(self, x):
        return self.network(x)
