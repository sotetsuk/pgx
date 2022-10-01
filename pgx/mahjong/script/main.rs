use std::io::Write;

fn search(x: usize, y: usize, arr: &mut Vec<usize>) {
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
                    arr[code >> 4] |= 0b01 << (2 * (code & 0b1111));
                }
            }
        } else {
            for head in heads.iter() {
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
                        arr[code >> 4] |= 0b10 << (2 * (code & 0b1111));
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
    let mut arr = vec![0; (16329 >> 4) + 1];
    for x in 0..5 {
        for y in 0..2 {
            search(x, y, &mut arr);
        }
    }

    let mut f = std::fs::File::create("cache_suited.json").unwrap();
    f.write_all(format!("{:?}", &arr).as_bytes()).unwrap();
}
