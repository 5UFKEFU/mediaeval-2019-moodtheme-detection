#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
整理音乐文件：
1. 递归扫描 src_dir，读取 Artist / Album / Title 标签
2. 清洗标签内容，去除网址等无关信息
3. 将文件按照 dst_dir/Artist/Album/Artist - Title.ext 保存
4. 生成 result.xlsx 记录整理结果
"""

import os
import re
import shutil
import argparse
from pathlib import Path
from typing import Tuple, Optional, List, Dict
import hashlib

import pandas as pd
from mutagen import File as MutagenFile
from mutagen.id3 import ID3, TIT2, TALB, TPE1, ID3NoHeaderError

# ------------------------------------------------------------
# 正则：检测并移除常见网址/多余前后缀
URL_RE = re.compile(r'https?://\S+|www\.\S+|\[.*?]|【.*?】', re.IGNORECASE)

# 若你有 "合法歌手名单" 文件，可在此加载，用于简单矫正
def load_artist_whitelist(path: Optional[str]) -> set:
    if not path or not Path(path).is_file():
        return set()
    with open(path, 'r', encoding='utf-8') as f:
        return {line.strip() for line in f if line.strip()}

# 清洗单个标签
def clean(text: str) -> str:
    if not text:
        return ''
    text = URL_RE.sub('', text)          # 去掉网址 / 广告
    text = text.replace('_', ' ')        # _ -> 空格
    text = re.sub(r'\s{2,}', ' ', text)  # 多空格压缩
    text = text.strip('-_ .')
    return text.strip()

# 读取音频文件标签
def read_tags(fp: Path) -> Tuple[str, str, str]:
    artist = album = title = ''
    try:
        audio = MutagenFile(fp)
        if audio is None:
            return artist, album, title
        # ID3 / MP4 / FLAC 都尽量兼容
        tags = audio.tags or {}
        # mutagen 的键有多种可能
        artist = (tags.get('artist') or tags.get('TPE1') or tags.get('©ART') or [''])[0]
        album  = (tags.get('album')  or tags.get('TALB') or tags.get('©alb') or [''])[0]
        title  = (tags.get('title')  or tags.get('TIT2') or tags.get('©nam') or [''])[0]
    except Exception:
        pass
    return map(clean, (artist, album, title))

# 写回已清洗的标签（可选，避免播放器继续显示脏数据）
def write_tags(fp: Path, artist: str, album: str, title: str):
    try:
        audio = ID3(fp)
    except ID3NoHeaderError:
        audio = ID3()
    audio.delall('TPE1'); audio.add(TPE1(encoding=3, text=artist))
    audio.delall('TALB'); audio.add(TALB(encoding=3, text=album))
    audio.delall('TIT2'); audio.add(TIT2(encoding=3, text=title))
    audio.save(fp)

# 生成目标路径
def build_dst_path(dst_root: Path, artist: str, album: str, title: str, ext: str) -> Path:
    safe = lambda s: re.sub(r'[\\/:*?"<>|]', '_', s) or 'Unknown'
    artist_dir = (safe(artist) or 'Unknown Artist')[:40]
    album_dir  = (safe(album)  or 'Unknown Album')[:40]
    filename   = f"{safe(artist)[:40]} - {safe(title)[:40]}{ext}"
    path = dst_root / artist_dir / album_dir / filename
    # 如果路径过长，使用哈希
    if len(str(path)) > 240:
        hashname = hashlib.md5(f"{artist}-{album}-{title}".encode('utf-8')).hexdigest()
        filename = f"{hashname}{ext}"
        path = dst_root / artist_dir / album_dir / filename
    return path

def get_bitrate(fp: Path) -> int:
    """获取音频比特率，获取不到返回0"""
    try:
        audio = MutagenFile(fp)
        if hasattr(audio.info, 'bitrate'):
            return audio.info.bitrate
    except Exception:
        pass
    return 0

# 核心流程
def organize(src_dir: Path, dst_dir: Path, whitelist: set) -> List[Dict]:
    log: List[Dict] = []
    for root, _, files in os.walk(src_dir):
        for name in files:
            fp = Path(root) / name
            ext = fp.suffix.lower()
            if ext not in {'.mp3', '.flac', '.m4a', '.wav'}:
                continue

            artist, album, title = read_tags(fp)

            # 有白名单时，若 artist 含网址或垃圾并且能匹配到白名单，则矫正
            if whitelist:
                for a in whitelist:
                    if a.lower() in artist.lower():
                        artist = a
                        break

            has_meta = artist and title
            record = {
                'original_path': str(fp),
                'artist': artist,
                'album': album,
                'title': title,
                'status': 'moved' if has_meta else 'unmoved',
                'new_path': '',
                'quality_compare': ''
            }

            if has_meta:
                dst_path = build_dst_path(dst_dir, artist, album, title, ext)
                dst_path.parent.mkdir(parents=True, exist_ok=True)
                if dst_path.exists():
                    # 比较音质
                    old_bitrate = get_bitrate(dst_path)
                    new_bitrate = get_bitrate(fp)
                    if new_bitrate > old_bitrate or (new_bitrate == old_bitrate and fp.stat().st_size > dst_path.stat().st_size):
                        # 新文件更好，覆盖旧文件
                        try:
                            os.remove(dst_path)
                            shutil.move(str(fp), dst_path)
                            write_tags(dst_path, artist, album, title)
                            record['new_path'] = str(dst_path)
                            record['quality_compare'] = f'覆盖旧文件（新比特率{new_bitrate} > 旧比特率{old_bitrate}）'
                        except Exception as e:
                            record['status'] = f'error: {e}'
                    else:
                        # 旧文件更好，删除新文件
                        record['status'] = 'skipped'
                        record['quality_compare'] = f'保留旧文件（新比特率{new_bitrate} <= 旧比特率{old_bitrate}）'
                        try:
                            os.remove(fp)
                        except Exception as e:
                            record['quality_compare'] += f'，但删除新文件失败: {e}'
                else:
                    try:
                        shutil.move(str(fp), dst_path)
                        write_tags(dst_path, artist, album, title)
                        record['new_path'] = str(dst_path)
                    except Exception as e:
                        record['status'] = f'error: {e}'
            log.append(record)
    return log

# 保存结果表格
def save_report(records: List[Dict], out_path: Path):
    # 如果文件已存在,读取现有数据
    existing_records = []
    if out_path.exists():
        try:
            existing_df = pd.read_excel(out_path)
            existing_records = existing_df.to_dict('records')
        except Exception as e:
            print(f"读取现有报表出错: {e}")
    
    # 合并新旧记录
    all_records = existing_records + records
    
    # 保存合并后的数据
    df = pd.DataFrame(all_records)
    df.to_excel(out_path, index=False)

# ----------------------------- CLI -----------------------------
def parse_args():
    p = argparse.ArgumentParser(description='音乐文件整理脚本')
    p.add_argument('src', help='待整理的源目录')
    p.add_argument('dst', help='整理后的目标目录')
    p.add_argument('-w', '--whitelist',
                   help='歌手白名单文本文件（可选，每行一个歌手名）')
    p.add_argument('-o', '--output', default='result.xlsx',
                   help='整理结果报表文件名 (默认: result.xlsx)')
    return p.parse_args()

def main():
    args = parse_args()
    src_dir = Path(args.src).expanduser().resolve()
    dst_dir = Path(args.dst).expanduser().resolve()
    whitelist = load_artist_whitelist(args.whitelist)
    print(f"Scanning {src_dir} ...")
    records = organize(src_dir, dst_dir, whitelist)
    out_path = Path(args.output).expanduser().resolve()
    save_report(records, out_path)
    print(f"Done! 报表已保存到 {out_path}")

if __name__ == '__main__':
    main()