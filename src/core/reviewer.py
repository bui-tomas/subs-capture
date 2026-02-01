import json
import os
import glob
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
from pypinyin import lazy_pinyin, Style
from dotenv import load_dotenv

load_dotenv()

SUBS_PATH = os.getenv('SUBS_PATH')


class SubtitleCorrector:
    def __init__(self, screenshots_folder: str, subs_folder: str):
        self.screenshots_folder = screenshots_folder
        self.subs_folder = subs_folder
        self.json_path = None
        self.output_path = None
        self.data = None
        self.subtitles = []
        self.current_idx = 0
        self.reviewed = set()
        self.screenshot_map = {}
        self.current_screenshots = []
        self.selected_screenshot_idx = 0
        
        self.root = tk.Tk()
        self.root.title('Subtitle Corrector')
        self.root.geometry('1600x850')
        
        self._setup_ui()
        self._populate_file_list()
    
    def _setup_ui(self):
        main_frame = ttk.Frame(self.root, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Screenshot panel (thumbnails + main + file selector)
        screenshot_panel = ttk.LabelFrame(main_frame, text='Screenshot', padding=5)
        screenshot_panel.pack(fill=tk.X, pady=(0, 10))
        
        # Thumbnails on left (scrollable)
        thumb_outer = ttk.Frame(screenshot_panel)
        thumb_outer.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        
        ttk.Label(thumb_outer, text='Offsets', font=('Arial', 9, 'bold')).pack()
        
        thumb_canvas = tk.Canvas(thumb_outer, width=130, height=400)
        thumb_scrollbar = ttk.Scrollbar(thumb_outer, orient='vertical', command=thumb_canvas.yview)
        thumb_canvas.configure(yscrollcommand=thumb_scrollbar.set)
        
        thumb_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        thumb_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        self.thumb_container = ttk.Frame(thumb_canvas)
        thumb_canvas.create_window((0, 0), window=self.thumb_container, anchor='nw')
        self.thumb_container.bind('<Configure>', lambda e: thumb_canvas.configure(scrollregion=thumb_canvas.bbox('all')))
        self.thumb_canvas = thumb_canvas
        
        self.thumbnail_labels = []
        
        # Main screenshot in center
        self.screenshot_label = ttk.Label(screenshot_panel, text='Select a file to begin')
        self.screenshot_label.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, anchor='center')
        
        # File selector on right
        file_panel = ttk.LabelFrame(screenshot_panel, text='Files', padding=5)
        file_panel.pack(side=tk.RIGHT, fill=tk.Y, padx=(10, 0))
        
        file_canvas = tk.Canvas(file_panel, width=250, height=350)
        file_scrollbar = ttk.Scrollbar(file_panel, orient='vertical', command=file_canvas.yview)
        file_canvas.configure(yscrollcommand=file_scrollbar.set)
        
        file_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        file_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        self.file_container = ttk.Frame(file_canvas)
        file_canvas.create_window((0, 0), window=self.file_container, anchor='nw')
        self.file_container.bind('<Configure>', lambda e: file_canvas.configure(scrollregion=file_canvas.bbox('all')))
        self.file_canvas = file_canvas
        
        self.selected_file = tk.StringVar()
        
        ttk.Button(file_panel, text='Load', command=self._load_selected_file).pack(side=tk.BOTTOM, pady=(10, 0))
        
        # Info bar
        info_bar = ttk.Frame(main_frame)
        info_bar.pack(fill=tk.X, pady=(0, 10))
        
        self.time_var = tk.StringVar()
        ttk.Label(info_bar, textvariable=self.time_var, font=('Courier', 11, 'bold')).pack(side=tk.LEFT)
        
        # Bottom panels (variants + composition)
        bottom_frame = ttk.Frame(main_frame)
        bottom_frame.pack(fill=tk.BOTH, expand=True)
        
        # Left: OCR Variants (scrollable)
        variants_panel = ttk.LabelFrame(bottom_frame, text='OCR Variants (click to append)', padding=10)
        variants_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        
        variants_canvas = tk.Canvas(variants_panel, height=150)
        variants_scrollbar = ttk.Scrollbar(variants_panel, orient='vertical', command=variants_canvas.yview)
        variants_canvas.configure(yscrollcommand=variants_scrollbar.set)
        
        variants_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        variants_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        self.variant_frame = ttk.Frame(variants_canvas)
        variants_canvas.create_window((0, 0), window=self.variant_frame, anchor='nw')
        self.variant_frame.bind('<Configure>', lambda e: variants_canvas.configure(scrollregion=variants_canvas.bbox('all')))
        self.variants_canvas = variants_canvas
        
        # Right: Composition
        comp_panel = ttk.LabelFrame(bottom_frame, text='Composition', padding=10)
        comp_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Hanzi entry + clear button
        entry_frame = ttk.Frame(comp_panel)
        entry_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.hanzi_entry = ttk.Entry(entry_frame, font=('Arial', 16), width=30)
        self.hanzi_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.hanzi_entry.bind('<KeyRelease>', self._update_preview)
        
        ttk.Button(entry_frame, text='✕', width=3, command=self._clear_entry).pack(side=tk.LEFT, padx=(5, 0))
        
        # Preview
        ttk.Label(comp_panel, text='Pinyin:', font=('Arial', 10)).pack(anchor=tk.W)
        self.preview_pinyin = tk.StringVar()
        ttk.Label(comp_panel, textvariable=self.preview_pinyin, font=('Arial', 14)).pack(anchor=tk.W)

        ttk.Label(comp_panel, text='English:', font=('Arial', 10)).pack(anchor=tk.W, pady=(10, 0))
        self.english_var = tk.StringVar()
        ttk.Label(comp_panel, textvariable=self.english_var, font=('Arial', 12), wraplength=400).pack(anchor=tk.W)
        
        ttk.Separator(comp_panel, orient='horizontal').pack(fill=tk.X, pady=10)
        
        # Current values
        ttk.Label(comp_panel, text='Current saved:', font=('Arial', 9, 'bold')).pack(anchor=tk.W)
        self.current_hanzi_var = tk.StringVar()
        self.current_pinyin_var = tk.StringVar()
        ttk.Label(comp_panel, textvariable=self.current_hanzi_var, font=('Arial', 11), foreground='gray').pack(anchor=tk.W)
        ttk.Label(comp_panel, textvariable=self.current_pinyin_var, font=('Arial', 10), foreground='gray').pack(anchor=tk.W)
        
        # Navigation bar
        nav_frame = ttk.Frame(main_frame)
        nav_frame.pack(fill=tk.X, pady=(10, 0))
        
        ttk.Button(nav_frame, text='← Previous', command=self.previous).pack(side=tk.LEFT, padx=5)
        ttk.Button(nav_frame, text='Skip', command=self.skip).pack(side=tk.LEFT, padx=5)
        ttk.Button(nav_frame, text='Save & Next →', command=self.save_and_next).pack(side=tk.LEFT, padx=5)
        ttk.Button(nav_frame, text='Jump to...', command=self.jump_to).pack(side=tk.LEFT, padx=5)
        
        filter_frame = ttk.Frame(nav_frame)
        filter_frame.pack(side=tk.RIGHT, padx=10)

        self.filter_nonlyrics = tk.BooleanVar(value=True)
        self.filter_empty = tk.BooleanVar(value=False)
        self.filter_multivariant = tk.BooleanVar(value=False)
        self.filter_singlevariant = tk.BooleanVar(value=False)
        self.filter_unchecked = tk.BooleanVar(value=False)

        ttk.Checkbutton(filter_frame, text='Non-lyrics', variable=self.filter_nonlyrics).pack(side=tk.LEFT, padx=2)
        ttk.Checkbutton(filter_frame, text='Empty', variable=self.filter_empty).pack(side=tk.LEFT, padx=2)
        ttk.Checkbutton(filter_frame, text='Multi-variant', variable=self.filter_multivariant).pack(side=tk.LEFT, padx=2)
        ttk.Checkbutton(filter_frame, text='Single-variant', variable=self.filter_singlevariant).pack(side=tk.LEFT, padx=2)
        ttk.Checkbutton(filter_frame, text='Unchecked', variable=self.filter_unchecked).pack(side=tk.LEFT, padx=2)
        
        self.progress_var = tk.StringVar()
        ttk.Label(nav_frame, textvariable=self.progress_var, font=('Arial', 10)).pack(side=tk.RIGHT, padx=20)
        
        # Status bar
        self.status_var = tk.StringVar(value='Select a file to begin')
        ttk.Label(main_frame, textvariable=self.status_var, font=('Arial', 9), foreground='gray').pack(anchor=tk.W, pady=(5, 0))
        
        # Click to unfocus
        self.root.bind('<Button-1>', self._handle_click)
    
    def _populate_file_list(self):
        '''Find all JSON files in subs folder'''
        for widget in self.file_container.winfo_children():
            widget.destroy()
        
        json_files = glob.glob(f'{self.subs_folder}/**/*.json', recursive=True)
        json_files.sort()
        
        for path in json_files:
            filename = os.path.basename(path)
            # Skip original subs files (non-processed)
            if not any(x in filename for x in ['_cleaned', '_reviewed']):
                continue
            
            rb = ttk.Radiobutton(
                self.file_container, 
                text=filename, 
                value=path, 
                variable=self.selected_file
            )
            rb.pack(anchor=tk.W, pady=1)
    
    def _load_selected_file(self):
        '''Load the selected JSON file'''
        path = self.selected_file.get()
        if not path:
            messagebox.showwarning('No file', 'Select a file first')
            return
        
        self.json_path = path
        
        # Determine output path
        if '_reviewed.json' in path:
            self.output_path = path
        elif '_cleaned.json' in path:
            self.output_path = path.replace('_cleaned.json', '_reviewed.json')
        elif '_raw.json' in path:
            self.output_path = path.replace('_raw.json', '_reviewed.json')
        else:
            self.output_path = path.replace('.json', '_reviewed.json')
        
        # Load data
        with open(path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        
        self.subtitles = self.data['subtitles']
        self.current_idx = 0
        self.reviewed = set()
        
        # Build screenshot map
        base_name = os.path.basename(path).replace('_reviewed.json', '').replace('_cleaned.json', '').replace('_raw.json', '').replace('.json', '')
        self.screenshots_folder = f'screenshots/{base_name}'
        self.screenshot_map = self._build_screenshot_map()
        
        self._find_first_unreviewed()
        self._bind_keys()
        self._load_current()
        
        self.status_var.set(f'Loaded: {os.path.basename(path)} | Enter/→/D = Save & Next | Space = Skip | ←/A = Previous')
        print(f'Loaded: {path}')
        print(f'Output: {self.output_path}')
        print(f'Screenshots: {self.screenshots_folder}')
    
    def _bind_keys(self):
        '''Bind navigation keys after file is loaded'''
        self.root.bind('<Return>', self._handle_return)
        self.root.bind('<space>', self._handle_space)
        self.root.bind('<Left>', lambda e: self.previous())
        self.root.bind('<Right>', lambda e: self.save_and_next())
        self.root.bind('<a>', self._handle_a)
        self.root.bind('<d>', self._handle_d)
    
    def _is_entry_focused(self):
        return self.root.focus_get() == self.hanzi_entry

    def _handle_return(self, event):
        if self._is_entry_focused():
            self.root.focus_set()
        self.save_and_next()

    def _handle_space(self, event):
        if not self._is_entry_focused():
            self.skip()

    def _handle_a(self, event):
        if not self._is_entry_focused():
            self.previous()

    def _handle_d(self, event):
        if not self._is_entry_focused():
            self.save_and_next()
    
    def _handle_click(self, event):
        widget = event.widget
        if widget != self.hanzi_entry:
            self.root.focus_set()
    
    def _build_screenshot_map(self) -> dict:
        '''Map start times to list of (offset, path) tuples'''
        screenshot_map = {}
        
        for img_path in glob.glob(f'{self.screenshots_folder}/sub_*.png'):
            filename = os.path.basename(img_path).replace('.png', '')
            parts = filename.split('_')
            start_time = float(parts[1].rstrip('s'))
            offset = float(parts[3].rstrip('s'))
            
            if start_time not in screenshot_map:
                screenshot_map[start_time] = []
            screenshot_map[start_time].append((offset, img_path))
        
        for start_time in screenshot_map:
            screenshot_map[start_time].sort(key=lambda x: x[0])
        
        return screenshot_map
    
    def _find_first_unreviewed(self):
        '''Start at first subtitle with empty hanzi'''
        for i, sub in enumerate(self.subtitles):
            if '♪' in sub.get('english', ''):
                continue
            if not sub.get('hanzi'):
                self.current_idx = i
                return
        self.current_idx = 0
    
    def _clear_entry(self):
        '''Clear the hanzi entry field'''
        self.hanzi_entry.delete(0, tk.END)
        self._update_preview()
    
    def _append_variant(self, text: str):
        '''Append variant text to entry field'''
        current = self.hanzi_entry.get()
        self.hanzi_entry.delete(0, tk.END)
        self.hanzi_entry.insert(0, current + text)
        self._update_preview()
    
    def _get_filtered_count(self):
        '''Count subtitles matching current filter'''
        return sum(1 for i in range(len(self.subtitles)) if self._matches_filter(i))
    
    def _load_current(self):
        '''Load current subtitle into UI'''
        if not self.subtitles:
            return
        
        sub = self.subtitles[self.current_idx]
        
        self.time_var.set(f'[{self.current_idx}] {sub["start"]:.2f}s → {sub["end"]:.2f}s')
        english = sub.get('english', '').replace('\n', ' ').strip()
        self.english_var.set(english)
        self.current_hanzi_var.set(sub.get('hanzi', '') or '(empty)')
        self.current_pinyin_var.set(sub.get('pinyin', '') or '(empty)')
        
        self._load_screenshots(sub['start'])
        
        # Clear and rebuild variants
        for widget in self.variant_frame.winfo_children():
            widget.destroy()
        
        metadata = sub.get('metadata', {})
        variants_str = metadata.get('variants', '')
        confidences = metadata.get('confidences', [])
        
        # Get adjacent subtitle variants and hanzi for comparison
        prev_variants = set()
        next_variants = set()
        prev_hanzi = ''
        next_hanzi = ''
        
        if self.current_idx > 0:
            prev_sub = self.subtitles[self.current_idx - 1]
            prev_meta = prev_sub.get('metadata', {})
            prev_str = prev_meta.get('variants', '')
            if prev_str:
                prev_variants = set(prev_str.split(';'))
            prev_hanzi = prev_sub.get('hanzi', '')
        
        if self.current_idx < len(self.subtitles) - 1:
            next_sub = self.subtitles[self.current_idx + 1]
            next_meta = next_sub.get('metadata', {})
            next_str = next_meta.get('variants', '')
            if next_str:
                next_variants = set(next_str.split(';'))
            next_hanzi = next_sub.get('hanzi', '')
        
        if variants_str:
            variants = variants_str.split(';')
            
            for i, variant in enumerate(variants):
                conf = confidences[i] if i < len(confidences) else 0.0
                
                frame = ttk.Frame(self.variant_frame)
                frame.pack(fill=tk.X, pady=3)
                
                # Check adjacency (substring match in either direction)
                in_prev = any(variant in pv or pv in variant for pv in prev_variants)
                in_next = any(variant in nv or nv in variant for nv in next_variants)
                
                # Check if variant is used in adjacent hanzi (saved composition)
                used_in_prev = prev_hanzi and (variant in prev_hanzi or prev_hanzi in variant)
                used_in_next = next_hanzi and (variant in next_hanzi or next_hanzi in variant)
                
                # Build indicators
                indicators = ''
                if in_prev:
                    indicators += '←'
                if used_in_prev:
                    indicators += '✗'
                if indicators and (in_next or used_in_next):
                    indicators += ' '
                if in_next:
                    indicators += '→'
                if used_in_next:
                    indicators += '✗'
                
                btn = ttk.Button(frame, text=variant, command=lambda v=variant: self._append_variant(v))
                btn.pack(side=tk.LEFT, padx=(0, 5))
                
                conf_text = f'({conf:.3f})'
                if indicators:
                    conf_text += f' {indicators}'
                
                # Color coding: red if used in adjacent hanzi, orange if just in variants
                if used_in_prev or used_in_next:
                    color = 'red'
                elif in_prev or in_next:
                    color = 'orange'
                else:
                    color = 'gray'
                
                ttk.Label(frame, text=conf_text, font=('Arial', 10), foreground=color).pack(side=tk.LEFT)
            
            # Add concat button at bottom of variants panel
            ttk.Separator(self.variant_frame, orient='horizontal').pack(fill=tk.X, pady=5)
            
            # Filter out variants that appear in adjacent hanzi
            clean_variants = []
            for v in variants:
                used_prev = prev_hanzi and (v in prev_hanzi or prev_hanzi in v)
                used_next = next_hanzi and (v in next_hanzi or next_hanzi in v)
                
                if not (used_prev or used_next):
                    clean_variants.append(v)
            
            def concat_clean():
                self.hanzi_entry.delete(0, tk.END)
                self.hanzi_entry.insert(0, ''.join(clean_variants))
                self._update_preview()
            
            concat_btn = ttk.Button(
                self.variant_frame, 
                text=f'Concat Clean ({len(clean_variants)}/{len(variants)})', 
                command=concat_clean
            )
            concat_btn.pack(pady=5)
        else:
            ttk.Label(self.variant_frame, text='No variants available', foreground='gray').pack()
        
        # Pre-populate entry with current hanzi
        self.hanzi_entry.delete(0, tk.END)
        if sub.get('hanzi'):
            self.hanzi_entry.insert(0, sub['hanzi'])
        self._update_preview()
        
        reviewed_count = len(self.reviewed)
        total = len(self.subtitles)
        filtered_count = self._get_filtered_count()
        self.progress_var.set(f'{reviewed_count}/{total} reviewed | {self.current_idx + 1}/{total} | {filtered_count} matching')
        self.root.focus_set()
        
    def _load_screenshots(self, start_time: float):
        '''Load all screenshots for given start time'''
        best_start = None
        best_diff = float('inf')
        
        for t in self.screenshot_map:
            diff = abs(t - start_time)
            if diff < best_diff:
                best_diff = diff
                best_start = t
        
        # Clear thumbnails
        for widget in self.thumb_container.winfo_children():
            widget.destroy()
        self.thumbnail_labels.clear()
        
        if best_start is None or best_diff > 1.0:
            self.current_screenshots = []
            self.screenshot_label.configure(image='', text='No screenshot available')
            ttk.Label(self.thumb_container, text='No screenshots', foreground='gray').pack()
            return
        
        self.current_screenshots = self.screenshot_map[best_start]
        self.selected_screenshot_idx = 0
        
        # Find center offset (closest to 0)
        for i, (offset, path) in enumerate(self.current_screenshots):
            if abs(offset) < 0.01:
                self.selected_screenshot_idx = i
                break
        
        # Create thumbnails
        for i, (offset, path) in enumerate(self.current_screenshots):
            try:
                img = Image.open(path)
                img.thumbnail((120, 70))
                photo = ImageTk.PhotoImage(img)
                
                frame = ttk.Frame(self.thumb_container)
                frame.pack(pady=3)
                
                label = ttk.Label(frame, image=photo, cursor='hand2')
                label.image = photo
                label.pack()
                
                offset_label = ttk.Label(frame, text=f'{offset:+.2f}s', font=('Arial', 8))
                offset_label.pack()
                
                label.bind('<Button-1>', lambda e, idx=i: self._select_thumbnail(idx))
                
                self.thumbnail_labels.append((label, frame))
            except Exception as e:
                print(f'Error loading thumbnail: {e}')
        
        self._show_main_screenshot()
        self._update_thumbnail_borders()
    
    def _select_thumbnail(self, idx: int):
        '''Select a thumbnail and show it enlarged'''
        self.selected_screenshot_idx = idx
        self._show_main_screenshot()
        self._update_thumbnail_borders()
    
    def _show_main_screenshot(self):
        '''Show currently selected screenshot in main view'''
        if not self.current_screenshots:
            return
        
        offset, path = self.current_screenshots[self.selected_screenshot_idx]
        
        try:
            img = Image.open(path)
            img.thumbnail((1000, 500))
            photo = ImageTk.PhotoImage(img)
            self.screenshot_label.configure(image=photo, text='')
            self.screenshot_label.image = photo
        except Exception as e:
            print(f'Error loading screenshot: {e}')
            self.screenshot_label.configure(image='', text='Error loading screenshot')
    
    def _update_thumbnail_borders(self):
        '''Highlight selected thumbnail'''
        for i, (label, frame) in enumerate(self.thumbnail_labels):
            if i == self.selected_screenshot_idx:
                frame.configure(relief='solid', borderwidth=2)
            else:
                frame.configure(relief='flat', borderwidth=0)
    
    def _update_preview(self, event=None):
        '''Update pinyin preview from entry field'''
        hanzi = self.hanzi_entry.get().strip()
        
        if hanzi:
            pinyin = ' '.join(lazy_pinyin(hanzi, style=Style.TONE))
            self.preview_pinyin.set(pinyin)
        else:
            self.preview_pinyin.set('(empty)')
    
    def save_and_next(self):
        '''Save current correction and move to next'''
        if not self.subtitles:
            return
        
        hanzi = self.hanzi_entry.get().strip()
        
        if hanzi:
            pinyin = ' '.join(lazy_pinyin(hanzi, style=Style.TONE))
            self.subtitles[self.current_idx]['hanzi'] = hanzi
            self.subtitles[self.current_idx]['pinyin'] = pinyin
            self.subtitles[self.current_idx]['manual_check'] = True
        
        self.reviewed.add(self.current_idx)
        self._save_json()
        self._move_next()
    
    def skip(self):
        '''Skip without saving changes'''
        if not self.subtitles:
            return
        self.reviewed.add(self.current_idx)
        self._move_next()
    
    def previous(self):
        '''Go to previous subtitle'''
        if not self.subtitles:
            return
        self._move_prev()
    
    def _move_next(self):
        '''Move to next subtitle based on filter'''
        for i in range(self.current_idx + 1, len(self.subtitles)):
            if self._matches_filter(i):
                self.current_idx = i
                self._load_current()
                return
        
        messagebox.showinfo('Done', 'Reached end of subtitles!')
    
    def _move_prev(self):
        '''Move to previous subtitle based on filter'''
        for i in range(self.current_idx - 1, -1, -1):
            if self._matches_filter(i):
                self.current_idx = i
                self._load_current()
                return
        
        messagebox.showinfo('Start', 'At the beginning!')
    
    def _matches_filter(self, idx: int) -> bool:
        '''Check if subtitle at idx matches all active filters'''
        sub = self.subtitles[idx]
        
        # If no filters selected, show all
        if not any([self.filter_nonlyrics.get(), self.filter_empty.get(), 
                    self.filter_multivariant.get(), self.filter_singlevariant.get(),
                    self.filter_unchecked.get()]):
            return True
        
        # Check lyrics
        is_lyrics = '♪' in sub.get('english', '')
        if self.filter_nonlyrics.get() and is_lyrics:
            return False
        
        # Check empty
        is_empty = not sub.get('hanzi')
        if self.filter_empty.get() and not is_empty:
            return False
        
        # Get variant count
        metadata = sub.get('metadata', {})
        variants_str = metadata.get('variants', '')
        variant_count = len(variants_str.split(';')) if variants_str else 0
        
        # Check multi-variant
        if self.filter_multivariant.get() and variant_count <= 1:
            return False
        
        # Check single-variant
        if self.filter_singlevariant.get() and variant_count != 1:
            return False
        
        # Check unchecked (not manually reviewed)
        if self.filter_unchecked.get() and sub.get('manual_check'):
            return False
        
        return True
        
    def jump_to(self):
        '''Jump to specific index'''
        if not self.subtitles:
            return
        
        dialog = tk.Toplevel(self.root)
        dialog.title('Jump to index')
        dialog.geometry('250x100')
        
        ttk.Label(dialog, text=f'Enter index (0-{len(self.subtitles)-1}):').pack(pady=10)
        entry = ttk.Entry(dialog)
        entry.pack()
        entry.focus()
        
        def do_jump():
            try:
                idx = int(entry.get())
                if 0 <= idx < len(self.subtitles):
                    self.current_idx = idx
                    self._load_current()
                    dialog.destroy()
                else:
                    messagebox.showerror('Error', 'Index out of range')
            except ValueError:
                messagebox.showerror('Error', 'Invalid number')
        
        ttk.Button(dialog, text='Jump', command=do_jump).pack(pady=10)
        entry.bind('<Return>', lambda e: do_jump())
    
    def _save_json(self):
        '''Save current state to output file'''
        if not self.data or not self.output_path:
            return
        
        self.data['subtitles'] = self.subtitles
        
        with open(self.output_path, 'w', encoding='utf-8') as f:
            json.dump(self.data, f, ensure_ascii=False, indent=2)
    
    def run(self):
        '''Start the GUI'''
        self.root.mainloop()


if __name__ == '__main__':
    base_name = SUBS_PATH.split('/')[-1].split('.')[0]
    subs_folder = f'subs/{SUBS_PATH.split("/")[1]}'
    screenshots_folder = f'screenshots/{base_name}'
    
    print(f'Subs folder: {subs_folder}')
    
    corrector = SubtitleCorrector(screenshots_folder, subs_folder)
    corrector.run()