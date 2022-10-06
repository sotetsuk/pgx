use std::io::Write;
use std::collections::BTreeSet;

fn search(x: usize, y: usize, arr: &mut Vec<BTreeSet<(usize, usize, usize, usize, usize)>>) {
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
                let mut zero = false;
                let mut valid = true;
                for i in 0..9 {
                    if hand[i] == 0 {
                        zero = true;
                    } else {
                        if zero {
                            valid = false;
                            break;
                        }
                        code <<= 1;
                        code += 1;
                        code <<= hand[i] - 1;
                    }
                }

                if valid {
                    let mut chow = 0;
                    let mut pung = 0;
                    for i in 0..x {
                        if sid[i] < 7 {
                            chow |= 1 << sid[i];
                        } else {
                            pung |= 1 << sid[i] - 7;
                        }
                    }
                    let mut double_chows = 0;
                    let mut used = 0;
                    for j in 0..x {
                        if (used >> sid[j] & 1) == 1 {
                            double_chows += 1;
                            used ^= 1 << sid[j];
                        } else {
                            used ^= 1 << sid[j];
                        }
                    }

                    let outside = match code {
                        0b100010001000 | 0b100100100 => if pung == 0 { 0b11 } else { 0 },
                        0b101010 | 0b111 | 0b100 => 0b11,
                        0b100011 => 0b01,
                        0b111000 => 0b10,
                        _ => 0b00,
                    };

                    arr[code].insert((9, chow, pung, double_chows, outside));
                }
            }
        } else {
            for (head_idx, head) in heads.iter().enumerate() {
                let hand: Vec<usize> = (0..9)
                    .map(|i| (0..x).map(|j| sets[sid[j]][i]).sum::<usize>() + head[i])
                    .collect();
                if hand.iter().all(|&h| h <= 4) {
                    let mut code = 0;
                    let mut zero = false;
                    let mut valid = true;
                    for i in 0..9 {
                        if hand[i] == 0 {
                            zero = true;
                        } else {
                            if zero {
                                valid = false;
                                break;
                            }
                            code <<= 1;
                            code += 1;
                            code <<= hand[i] - 1;
                        }
                    }

                    if valid {
                        let mut chow = 0;
                        let mut pung = 0;
                        for i in 0..x {
                            if sid[i] < 7 {
                                chow |= 1 << sid[i];
                            } else {
                                pung |= 1 << sid[i] - 7;
                            }
                        }
                        let mut double_chows = 0;
                        let mut used = 0;
                        for j in 0..x {
                            if (used >> sid[j] & 1) == 1 {
                                double_chows += 1;
                                used ^= 1 << sid[j];
                            } else {
                                used ^= 1 << sid[j];
                            }
                        }

                        // 11112233m (only left)
                        // 11223333m (only right)
                        // 11123m (only left)
                        // 12333m (only right)
                        // 11m
                        let outside = match code {
                            0b10 => 0b11,
                            0b10001010 | 0b10011 => 0b01,
                            0b10101000 | 0b11100 => 0b10,
                            _ => 0b00,
                        };

                        arr[code].insert((head_idx, chow, pung, double_chows, outside));
                    }
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
    let mut arr = vec![BTreeSet::new(); 16329];
    for x in 0..5 {
        for y in 0..2 {
            search(x, y, &mut arr);
        }
    }

    let mut cache = vec![vec![0; 3]; 16329];

    for (idx,vs) in arr.iter().enumerate() {
        if vs.is_empty() {
            continue;
        }
        let vs: Vec<_> = vs.iter().collect();
        for i in 0..3 {
            let &(h, chow, pung, double_chows, outside) = vs[i % vs.len()];
            cache[idx][i] = h | (chow << 4) | (pung << 11) | (double_chows << 20) | (outside << 22);
        }
    }

    let mut f = std::fs::File::create("yaku_cache.json").unwrap();
    f.write_all(format!("{:?}", &cache).as_bytes()).unwrap();
}
