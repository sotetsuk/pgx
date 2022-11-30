#! python3

"""
* すずめ雀の牌は11桁の0~4の数字で表現できる（5進数11桁）
* 5進数11桁は最大値48,828,124なのでint32で表現可能
"""

print(int("44444444444", 5))

shunzu = [
    [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0],
]

kouzu = [
    [3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3],
]

win_hands = {}
for i, x in enumerate((shunzu, kouzu)):
    for j, y in enumerate((shunzu, kouzu)):
        for a in x:
            for b in y:
                combined = [e1 + e2 for e1, e2 in zip(a, b)]
                ok = True
                for e in combined:
                    if e > 4:
                        ok = False
                if not ok:
                    break
                base_score = 0
                base_score += 1 if i == 0 else 2
                base_score += 1 if j == 0 else 2
                is_tanyao = True
                is_chanta = True
                is_tinyao = True
                is_all_green = True
                for c in (a, b):
                    for k, e in enumerate(c):
                        if k == 0 and e > 0:  # 1s
                            is_tanyao = False
                            is_all_green = False
                        elif k == 1 and e > 0:  # 2s
                            is_chanta = False
                            is_tinyao = False
                        elif k == 2 and e > 0:  # 3s
                            is_chanta = False
                            is_tinyao = False
                        elif k == 3 and e > 0:  # 4s
                            is_chanta = False
                            is_tinyao = False
                        elif k == 4 and e > 0:  # 5s
                            is_chanta = False
                            is_tinyao = False
                            is_all_green = False
                        elif k == 5 and e > 0:  # 6s
                            is_chanta = False
                            is_tinyao = False
                        elif k == 6 and e > 0:  # 7s
                            is_chanta = False
                            is_tinyao = False
                            is_all_green = False
                        elif k == 7 and e > 0:  # 8s
                            is_chanta = False
                            is_tinyao = False
                        elif k == 8 and e > 0:  # 9s
                            is_tanyao = False
                            is_all_green = False
                        elif k == 9 and e > 0:  # gd
                            is_tanyao = False
                        elif k == 10 and e > 0:  # rd
                            is_tanyao = False
                            is_all_green = False

                yaku_score = 0
                if is_all_green or is_tinyao:  # yakuman
                    yaku_score += 10 if is_all_green else 15
                else:
                    if is_tanyao:
                        yaku_score += 1
                    if is_chanta:
                        yaku_score += 2

                key = "".join([str(e) for e in combined])
                win_hands[key] = (base_score, yaku_score)


win_hands = list(win_hands.items())
win_hands.sort()


print(len(win_hands))


def to_base5(hand):
    return int(hand, 5)


for i, (k, v) in enumerate(win_hands):
    print(f"{i:3d}\t{to_base5(k):8d}\t{k}\t{v[0]}\t{v[1]:2d}")

print([to_base5(k) for k, _ in win_hands])
print([v[0] for _, v in win_hands])
print([v[1] for _, v in win_hands])
