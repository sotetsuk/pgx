use std::collections::{BTreeSet, VecDeque};
use std::io::Write;

fn encode(hand: &Vec<usize>) -> usize {
    let mut code: usize = 0;
    for h in hand.iter() {
        code = 5 * code + h;
    }
    code
}

fn complete_hands_suited(x: usize, y: usize) -> BTreeSet<Vec<usize>> {
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

    let mut ret: BTreeSet<Vec<usize>> = BTreeSet::new();

    let mut sid = vec![0; x + 1];

    while sid[x] == 0 {
        if y == 0 {
            let hand: Vec<usize> = (0..9)
                .map(|i| (0..x).map(|j| sets[sid[j]][i]).sum::<usize>())
                .collect();
            if hand.iter().all(|&h| h <= 4) {
                ret.insert(hand);
            }
        } else {
            for head in heads.iter() {
                let hand: Vec<usize> = (0..9)
                    .map(|i| (0..x).map(|j| sets[sid[j]][i]).sum::<usize>() + head[i])
                    .collect();
                if hand.iter().all(|&h| h <= 4) {
                    ret.insert(hand);
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

    ret
}

fn bfs_suited(ws: BTreeSet<Vec<usize>>) -> Vec<usize> {
    let mut dist = vec![usize::MAX; 1953125];
    let mut deq: VecDeque<(usize, Vec<usize>)> = VecDeque::new();

    for hand in ws {
        dist[encode(&hand)] = 0;
        deq.push_back((0, hand));
    }

    while !deq.is_empty() {
        let (d, hand) = deq.pop_front().unwrap();
        if d > dist[encode(&hand)] {
            continue;
        }
        for k in 0..9 {
            if hand[k] < 4 {
                let mut hand_add = hand.clone();
                hand_add[k] += 1;
                let code_add = encode(&hand_add);
                if dist[code_add] > d {
                    dist[code_add] = d;
                    deq.push_front((d, hand_add));
                }
            }
            if hand[k] > 0 {
                let mut hand_sub = hand.clone();
                hand_sub[k] -= 1;
                let code_sub = encode(&hand_sub);
                if dist[code_sub] > d + 1 {
                    dist[code_sub] = d + 1;
                    deq.push_back((d + 1, hand_sub));
                }
            }
        }
    }

    dist
}

fn complete_hands_honor(x: usize, y: usize) -> BTreeSet<Vec<usize>> {
    assert!(x <= 4);
    assert!(y <= 1);
    let sets = vec![
        vec![3, 0, 0, 0, 0, 0, 0],
        vec![0, 3, 0, 0, 0, 0, 0],
        vec![0, 0, 3, 0, 0, 0, 0],
        vec![0, 0, 0, 3, 0, 0, 0],
        vec![0, 0, 0, 0, 3, 0, 0],
        vec![0, 0, 0, 0, 0, 3, 0],
        vec![0, 0, 0, 0, 0, 0, 3],
    ];

    let heads = vec![
        vec![2, 0, 0, 0, 0, 0, 0],
        vec![0, 2, 0, 0, 0, 0, 0],
        vec![0, 0, 2, 0, 0, 0, 0],
        vec![0, 0, 0, 2, 0, 0, 0],
        vec![0, 0, 0, 0, 2, 0, 0],
        vec![0, 0, 0, 0, 0, 2, 0],
        vec![0, 0, 0, 0, 0, 0, 2],
    ];

    let mut ret: BTreeSet<Vec<usize>> = BTreeSet::new();

    let mut sid = vec![0; x + 1];

    while sid[x] == 0 {
        if y == 0 {
            let hand: Vec<usize> = (0..7)
                .map(|i| (0..x).map(|j| sets[sid[j]][i]).sum::<usize>())
                .collect();
            if hand.iter().all(|&h| h <= 4) {
                ret.insert(hand);
            }
        } else {
            for head in heads.iter() {
                let hand: Vec<usize> = (0..7)
                    .map(|i| (0..x).map(|j| sets[sid[j]][i]).sum::<usize>() + head[i])
                    .collect();
                if hand.iter().all(|&h| h <= 4) {
                    ret.insert(hand);
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

    ret
}

fn bfs_honor(ws: BTreeSet<Vec<usize>>) -> Vec<usize> {
    let mut dist = vec![usize::MAX; 78125];
    let mut deq: VecDeque<(usize, Vec<usize>)> = VecDeque::new();

    for hand in ws {
        dist[encode(&hand)] = 0;
        deq.push_back((0, hand));
    }

    while !deq.is_empty() {
        let (d, hand) = deq.pop_front().unwrap();
        if d > dist[encode(&hand)] {
            continue;
        }
        for k in 0..7 {
            if hand[k] < 4 {
                let mut hand_add = hand.clone();
                hand_add[k] += 1;
                let code_add = encode(&hand_add);
                if dist[code_add] > d {
                    dist[code_add] = d;
                    deq.push_front((d, hand_add));
                }
            }
            if hand[k] > 0 {
                let mut hand_sub = hand.clone();
                hand_sub[k] -= 1;
                let code_sub = encode(&hand_sub);
                if dist[code_sub] > d + 1 {
                    dist[code_sub] = d + 1;
                    deq.push_back((d + 1, hand_sub));
                }
            }
        }
    }

    dist
}

fn main() {
    let mut arr = vec![vec![0; 9]; 1953125 + 78125];
    for y in 0..2 {
        for x in 0..5 {
            if x == 0 && y == 0 {
                continue;
            }
            let ws = complete_hands_suited(x, y);
            let dist = bfs_suited(ws);
            for code in 0..1953125 {
                arr[code][y * 5 + x - 1] = dist[code];
            }

            let ws = complete_hands_honor(x, y);
            let dist = bfs_honor(ws);
            for code in 0..78125 {
                arr[code + 1953125][y * 5 + x - 1] = dist[code];
            }
        }
    }

    for code in 0..1953125 + 78125 {
        for x in (0..3).rev() {
            arr[code][x + 1] -= arr[code][x];
        }
        for x in (4..8).rev() {
            arr[code][x + 1] -= arr[code][x];
        }
    }

    let mut f = std::fs::File::create("shanten_cache.json").unwrap();
    f.write_all(format!("{:?}", &arr).as_bytes()).unwrap();
}
