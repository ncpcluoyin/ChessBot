"""
统计 self-play PGN 中的王车易位数量。
"""

import glob, os, sys, collections

def count_castling(pgn_dir="data/self_play_games"):
    files = sorted(glob.glob(os.path.join(pgn_dir, "*.pgn")))
    if not files:
        print(f"No PGN files found in {pgn_dir}")
        return

    total_games = 0
    castle = collections.Counter()

    for fp in files:
        with open(fp) as f:
            text = f.read()
        total_games += 1
        # PGN 中易位记为 O-O 和 O-O-O
        castle['O-O'] += text.count('O-O') - text.count('O-O-O')
        castle['O-O-O'] += text.count('O-O-O')

    print(f"检查 {total_games} 局 ({os.path.join(pgn_dir, '*.pgn')})")
    print(f"  短易位 O-O:   {castle['O-O']} 次 ({castle['O-O']/max(total_games,1):.2f}/局)")
    print(f"  长易位 O-O-O: {castle['O-O-O']} 次 ({castle['O-O-O']/max(total_games,1):.2f}/局)")
    print(f"  合计:          {castle['O-O']+castle['O-O-O']} 次")

if __name__ == '__main__':
    d = sys.argv[1] if len(sys.argv) > 1 else "data/self_play_games"
    count_castling(d)
