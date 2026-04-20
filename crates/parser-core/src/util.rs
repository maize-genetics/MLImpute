/// Natural sort comparison for chromosome names (chr1, chr2, chr10, etc.)
pub fn natural_sort(a: &str, b: &str) -> std::cmp::Ordering {
    let extract_num = |s: &str| -> Option<u32> {
        s.chars()
            .filter(|c| c.is_ascii_digit())
            .collect::<String>()
            .parse::<u32>()
            .ok()
    };

    match (extract_num(a), extract_num(b)) {
        (Some(na), Some(nb)) => na.cmp(&nb),
        _ => a.cmp(b),
    }
}
