IDX = []
# 縦
for i in range(3):
    for j in range(7):
        a = i * 7 + j
        IDX.append([a, a + 7, a + 14, a + 21])
# 横
for i in range(6):
    for j in range(4):
        a = i * 7 + j
        IDX.append([a, a + 1, a + 2, a + 3])

# 斜め
for i in range(3):
    for j in range(4):
        a = i * 7 + j
        IDX.append([a, a + 8, a + 16, a + 24])
for i in range(3):
    for j in range(3, 7):
        a = i * 7 + j
        IDX.append([a, a + 6, a + 12, a + 18])

print(IDX)
