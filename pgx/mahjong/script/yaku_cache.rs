use std::collections::BTreeSet;
use std::io::Write;

fn search(x: usize, y: usize, arr: &mut Vec<BTreeSet<(usize, usize, usize, usize, usize, usize)>>) {
    assert!(x <= 4);
    assert!(y <= 1);
    let sets = vec![
        vec![1, 1, 1, 0, 0, 0, 0, 0, 0],
        vec![0, 1, 1, 1, 0, 0, 0, 0, 0],
        vec![0, 0, 1, 1, 1, 0, 0, 0, 0],
        vec![0, 0, 0, 1, 1, 1, 0, 0, 0],
        vec![0, 0, 0, 0, 1, 1, 1, 0, 0],
        vec![0, 0, 0, 0, 0, 1, 1, 1, 0],
        vec![0, 0, 0, 0, 0, 0, 1, 1, 1],
        vec![3, 0, 0, 0, 0, 0, 0, 0, 0],
        vec![0, 3, 0, 0, 0, 0, 0, 0, 0],
        vec![0, 0, 3, 0, 0, 0, 0, 0, 0],
        vec![0, 0, 0, 3, 0, 0, 0, 0, 0],
        vec![0, 0, 0, 0, 3, 0, 0, 0, 0],
        vec![0, 0, 0, 0, 0, 3, 0, 0, 0],
        vec![0, 0, 0, 0, 0, 0, 3, 0, 0],
        vec![0, 0, 0, 0, 0, 0, 0, 3, 0],
        vec![0, 0, 0, 0, 0, 0, 0, 0, 3],
    ];

    let heads = vec![
        vec![2, 0, 0, 0, 0, 0, 0, 0, 0],
        vec![0, 2, 0, 0, 0, 0, 0, 0, 0],
        vec![0, 0, 2, 0, 0, 0, 0, 0, 0],
        vec![0, 0, 0, 2, 0, 0, 0, 0, 0],
        vec![0, 0, 0, 0, 2, 0, 0, 0, 0],
        vec![0, 0, 0, 0, 0, 2, 0, 0, 0],
        vec![0, 0, 0, 0, 0, 0, 2, 0, 0],
        vec![0, 0, 0, 0, 0, 0, 0, 2, 0],
        vec![0, 0, 0, 0, 0, 0, 0, 0, 2],
    ];

    let mut sid = vec![0; x + 1];

    while sid[x] == 0 {
        if y == 0 {
            let hand: Vec<usize> = (0..9)
                .map(|i| (0..x).map(|j| sets[sid[j]][i]).sum::<usize>())
                .collect();
            if hand.iter().all(|&h| h <= 4) {
                let mut code = 0;
                for i in 0..9 {
                    code = code * 5 + hand[i];
                }

                let mut chow = 0;
                let mut pung = 0;
                for i in 0..x {
                    if sid[i] < 7 {
                        chow |= 1 << sid[i];
                    } else {
                        pung |= 1 << sid[i] - 7;
                    }
                }
                let mut n_double_chow = 0;
                let mut used = 0;
                for j in 0..x {
                    if (used >> sid[j] & 1) == 1 {
                        n_double_chow += 1;
                        used ^= 1 << sid[j];
                    } else {
                        used ^= 1 << sid[j];
                    }
                }

                let mut outside = 1;
                for i in 0..x {
                    outside &= match sid[i] {
                        0 | 6 | 7 | 15 => 1,
                        _ => 0,
                    };
                }

                arr[code].insert((9, chow, pung, n_double_chow, outside, 0));
            }
        } else {
            for (head_idx, head) in heads.iter().enumerate() {
                let hand: Vec<usize> = (0..9)
                    .map(|i| (0..x).map(|j| sets[sid[j]][i]).sum::<usize>() + head[i])
                    .collect();
                if hand.iter().all(|&h| h <= 4) {
                    let mut code = 0;
                    for i in 0..9 {
                        code = code * 5 + hand[i];
                    }

                    let mut chow = 0;
                    let mut pung = 0;
                    for i in 0..x {
                        if sid[i] < 7 {
                            chow |= 1 << sid[i];
                        } else {
                            pung |= 1 << sid[i] - 7;
                        }
                    }
                    let mut n_double_chow = 0;
                    let mut used = 0;
                    for j in 0..x {
                        if (used >> sid[j] & 1) == 1 {
                            n_double_chow += 1;
                            used ^= 1 << sid[j];
                        } else {
                            used ^= 1 << sid[j];
                        }
                    }

                    let mut outside = match head_idx {
                        0 | 8 => 1,
                        _ => 0,
                    };
                    for i in 0..x {
                        outside &= match sid[i] {
                            0 | 6 | 7 | 15 => 1,
                            _ => 0,
                        };
                    }

                    let nine_gates = (hand[0] >= 3)
                        & (hand[1] >= 1)
                        & (hand[2] >= 1)
                        & (hand[3] >= 1)
                        & (hand[4] >= 1)
                        & (hand[5] >= 1)
                        & (hand[6] >= 1)
                        & (hand[7] >= 1)
                        & (hand[8] >= 3);
                    arr[code].insert((
                        head_idx,
                        chow,
                        pung,
                        n_double_chow,
                        outside,
                        nine_gates as usize,
                    ));
                }
            }
        }

        let mut i = 0;
        sid[i] += 1;
        while sid[i] == sets.len() {
            sid[i] = 0;
            i += 1;
            sid[i] += 1;
        }
    }
}

fn main() {
    let mut arr = vec![BTreeSet::new(); 1953125];
    for x in 0..5 {
        for y in 0..2 {
            search(x, y, &mut arr);
        }
    }

    let mut cache = vec![vec![0; 3]; 1953125];

    for (idx, vs) in arr.iter().enumerate() {
        if vs.is_empty() {
            continue;
        }
        let vs: Vec<_> = vs.iter().collect();
        for i in 0..3 {
            let &(h, chow, pung, n_double_chow, outside, nine_gates) = vs[i % vs.len()];
            cache[idx][i] = h
                | (chow << 4)
                | (pung << 11)
                | ((pung as u32).count_ones() << 20) as usize
                | (n_double_chow << 23)
                | (outside << 25)
                | (nine_gates << 26);
        }
    }

    let mut f = std::fs::File::create("yaku_cache.json").unwrap();
    f.write_all(format!("{:?}", &cache).as_bytes()).unwrap();
}
