import tkinter as tk
from tkinter import scrolledtext, messagebox, simpledialog, font as tkFont, ttk
import io
import sys
import traceback
import ctypes

try:
    from Lcompiler import Lexer, ParserAndGenerator, VM, SymbolTable, Symbol
except ImportError:
    messagebox.showerror("导入错误", "无法从 compiler.py 导入编译器组件。请确保文件位于同一目录或可通过 PYTHONPATH 访问。")
    sys.exit(1)
except Exception as e:
    messagebox.showerror("导入错误", f"导入过程中发生错误: {e}")
    sys.exit(1)

class CompilerGUI:
    def __init__(self, master):
        self.master = master
        master.title("编译器前端 v2.1")

        try:
            if sys.platform == "win32":
                ctypes.windll.shcore.SetProcessDpiAwareness(1)
        except Exception as e:
            print(f"Info: 未能设置DPI感知 (可能是非Windows系统或缺少库): {e}")

        self.current_font_size = 14
        self.font_family = "Consolas"
        if "Consolas" not in tkFont.families():
            self.font_family = "Courier New"
        if "Courier New" not in tkFont.families() and self.font_family == "Courier New":
            self.font_family = "Monospace"
        
        self.update_font_config()

        master.geometry("2160x1350")

        # --- 主框架 ---
        main_paned_window = tk.PanedWindow(master, orient=tk.HORIZONTAL, sashrelief=tk.RAISED, bd=2, relief=tk.SUNKEN)
        main_paned_window.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # --- 左侧容器框架 ---
        # 创建一个主框架来容纳左侧的所有组件 (按钮和可调整窗格)
        left_panel_container = tk.Frame(main_paned_window)
        main_paned_window.add(left_panel_container, stretch="always")

        # --- 右侧面板 (VM Code) ---
        right_panel = tk.Frame(main_paned_window)
        main_paned_window.add(right_panel, stretch="always")

        # 设置初始比例 (57.5 : 42.5)
        def set_initial_ratio():
            total_width = main_paned_window.winfo_width()
            if total_width > 100:
                main_paned_window.sash_place(0, int(total_width * 0.575), 0)
            else:
                self.master.after(100, set_initial_ratio)
        
        self.master.after(100, set_initial_ratio)
        
        # --- 控制按钮面板 ---
        # 将按钮放在左侧容器的底部
        controls_frame = tk.Frame(left_panel_container)
        controls_frame.pack(side=tk.BOTTOM, fill="x", pady=5)

        self.button_font = tkFont.Font(family=self.font_family, size=16, weight="bold")
        
        self.compile_button = tk.Button(controls_frame, text="编译并运行", command=self.compile_and_run, 
                                      font=self.button_font, bg="#4CAF50", fg="white")
        self.compile_button.pack(side=tk.LEFT, padx=10, pady=5)
        
        self.clear_button = tk.Button(controls_frame, text="清空", command=self.clear_all, 
                                    font=self.button_font, bg="#f44336", fg="white")
        self.clear_button.pack(side=tk.LEFT, padx=10, pady=5)
        
        tk.Label(controls_frame, text="字体:", font=self.ui_font).pack(side=tk.LEFT, padx=(20,0))
        self.increase_font_button = tk.Button(controls_frame, text="+", command=self.increase_font_size, 
                                            width=3, font=self.ui_font)
        self.increase_font_button.pack(side=tk.LEFT, padx=2)
        self.decrease_font_button = tk.Button(controls_frame, text="-", command=self.decrease_font_size, 
                                            width=3, font=self.ui_font)
        self.decrease_font_button.pack(side=tk.LEFT, padx=2)

        # --- 左侧可调整窗格 ---
        left_paned_window = tk.PanedWindow(left_panel_container, orient=tk.VERTICAL, sashrelief=tk.RAISED, bd=1, relief=tk.SUNKEN)
        left_paned_window.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # --- 左侧面板内容 ---
        # 1. 源代码输入 (添加到新的 PanedWindow)
        input_frame = tk.LabelFrame(left_paned_window, text="源代码", padx=5, pady=5, font=self.label_font)
        left_paned_window.add(input_frame, stretch="always")
        self.source_code_text = scrolledtext.ScrolledText(input_frame, wrap=tk.WORD, undo=True, font=self.content_font)
        self.source_code_text.pack(fill="both", expand=True)

        # 2. 符号表显示 (添加到新的 PanedWindow)
        symtable_frame = tk.LabelFrame(left_paned_window, text="符号表 & 结构体定义", padx=5, pady=5, font=self.label_font)
        left_paned_window.add(symtable_frame, stretch="always")
        
        symtable_container = tk.Frame(symtable_frame)
        symtable_container.pack(fill="both", expand=True)
        symtable_xscroll = tk.Scrollbar(symtable_container, orient=tk.HORIZONTAL)
        symtable_xscroll.pack(side=tk.BOTTOM, fill=tk.X)
        symtable_yscroll = tk.Scrollbar(symtable_container)
        symtable_yscroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.symtable_text = tk.Text(
            symtable_container, wrap=tk.NONE, state='disabled', font=self.content_font,
            xscrollcommand=symtable_xscroll.set, yscrollcommand=symtable_yscroll.set
        )
        self.symtable_text.pack(side=tk.LEFT, fill="both", expand=True)
        symtable_xscroll.config(command=self.symtable_text.xview)
        symtable_yscroll.config(command=self.symtable_text.yview)

        # 3. 状态/错误 和 ANS 输出 (添加到新的 PanedWindow)
        status_ans_notebook_frame = tk.Frame(left_paned_window)
        left_paned_window.add(status_ans_notebook_frame, stretch="always")
        
        self.status_ans_notebook = ttk.Notebook(status_ans_notebook_frame)
        
        status_tab_frame = tk.Frame(self.status_ans_notebook)
        status_inner_frame = tk.LabelFrame(status_tab_frame, text="状态 / 错误信息", padx=5, pady=5, font=self.label_font)
        status_inner_frame.pack(fill="both", expand=True, padx=2, pady=2)
        self.status_text = scrolledtext.ScrolledText(status_inner_frame, wrap=tk.WORD, foreground="red", font=self.content_font)
        self.status_text.pack(fill="both", expand=True)
        self.status_ans_notebook.add(status_tab_frame, text='状态/错误')

        ans_tab_frame = tk.Frame(self.status_ans_notebook)
        ans_inner_frame = tk.LabelFrame(ans_tab_frame, text="程序输出 (ANS)", padx=5, pady=5, font=self.label_font)
        ans_inner_frame.pack(fill="both", expand=True, padx=2, pady=2)
        self.ans_text = scrolledtext.ScrolledText(ans_inner_frame, wrap=tk.WORD, state='disabled', font=self.content_font)
        self.ans_text.pack(fill="both", expand=True)
        self.status_ans_notebook.add(ans_tab_frame, text='程序输出')
        
        self.status_ans_notebook.pack(fill="both", expand=True)

        # 设置左侧面板垂直比例
        def set_vertical_ratio():
            total_height = left_paned_window.winfo_height()
            if total_height > 100:
                # 源代码:40%, 符号表:30%, 状态/ANS:30%
                left_paned_window.sash_place(0, 0, int(total_height * 0.4))
                left_paned_window.sash_place(1, 0, int(total_height * 0.7))
            else:
                self.master.after(100, set_vertical_ratio)
        
        self.master.after(100, set_vertical_ratio)

        # --- 右侧面板内容 (VM Code) ---
        vm_code_frame = tk.LabelFrame(right_panel, text="生成的VM代码", padx=5, pady=5, font=self.label_font)
        vm_code_frame.pack(fill="both", expand=True, padx=5, pady=5)
        self.vm_code_text = scrolledtext.ScrolledText(vm_code_frame, wrap=tk.WORD, state='disabled', font=self.content_font)
        self.vm_code_text.pack(fill="both", expand=True)

        self.font_widgets = [
            self.source_code_text, self.vm_code_text, self.symtable_text,
            self.ans_text, self.status_text, self.compile_button, self.clear_button,
            self.increase_font_button, self.decrease_font_button
        ]
        self.label_frames_for_font = [
            input_frame, symtable_frame, vm_code_frame, 
            status_inner_frame, ans_inner_frame
        ]
        for child in controls_frame.winfo_children():
            if isinstance(child, tk.Label):
                self.font_widgets.append(child)
        
        self.apply_all_fonts()

    def update_font_config(self):
        self.content_font = tkFont.Font(family=self.font_family, size=self.current_font_size)
        self.label_font = tkFont.Font(family=self.font_family, size=self.current_font_size - 2 if self.current_font_size > 9 else self.current_font_size, weight="bold")
        self.ui_font = tkFont.Font(family="TkDefaultFont", size=self.current_font_size - 1 if self.current_font_size > 8 else self.current_font_size)
        self.button_font = tkFont.Font(family=self.font_family, size=16, weight="bold")

    def apply_all_fonts(self):
        self.update_font_config()
        for widget in self.font_widgets:
            try:
                if not widget.winfo_exists(): continue
                
                if isinstance(widget, (scrolledtext.ScrolledText, tk.Text)):
                    current_font_to_use = self.content_font
                elif isinstance(widget, (tk.Button, tk.Label)):
                    current_font_to_use = self.ui_font
                else:
                    current_font_to_use = self.ui_font
                
                widget.config(font=current_font_to_use)
            except tk.TclError:
                pass

        self.compile_button.config(font=self.button_font)
        self.clear_button.config(font=self.button_font)

        for lframe in self.label_frames_for_font:
            try:
                if not lframe.winfo_exists(): continue
                lframe.config(font=self.label_font)
            except tk.TclError:
                pass
        
        try:
            style = ttk.Style()
            current_theme = style.theme_use()
            if not current_theme:
                 pass

            try:
                style.configure("TNotebook.Tab", font=self.ui_font)
            except tk.TclError as e:
                print(f"Info: Could not configure Notebook tab font directly: {e}")
                if current_theme:
                    try:
                        style.theme_use(current_theme)
                    except tk.TclError:
                        pass
        except Exception as e:
            print(f"Info: General error during ttk style configuration: {e}")

    def increase_font_size(self):
        if self.current_font_size < 24:
            self.current_font_size += 1
            self.apply_all_fonts()

    def decrease_font_size(self):
        if self.current_font_size > 8:
            self.current_font_size -= 1
            self.apply_all_fonts()

    def _set_text_area_content(self, text_area, content):
        try:
            if not text_area.winfo_exists(): return
            
            is_scrolledtext = isinstance(text_area, scrolledtext.ScrolledText)
            target_text_widget = text_area if is_scrolledtext else text_area

            target_text_widget.config(state='normal')
            target_text_widget.delete(1.0, tk.END)
            target_text_widget.insert(tk.END, content)
            target_text_widget.see(tk.END)
            target_text_widget.config(state='disabled')
        except tk.TclError:
            pass

    def clear_all(self):
        # 对于 ScrolledText，需要在清除之前启用
        self.source_code_text.config(state='normal')
        self.source_code_text.delete(1.0, tk.END)
        self._set_text_area_content(self.vm_code_text, "")
        self._set_text_area_content(self.symtable_text, "")
        self._set_text_area_content(self.ans_text, "")

        self.status_text.config(state='normal')
        self.status_text.delete(1.0, tk.END)
        self.status_text.insert(tk.END, "已清空。")

        try:
            self.status_ans_notebook.select(0)
        except tk.TclError:
            pass

    def capture_symbol_table_display(self, symtable_obj):
        old_stdout = sys.stdout
        sys.stdout = captured_output = io.StringIO()
        try:
            symtable_obj.display()
        except Exception as e:
            return f"显示符号表时出错: {e}\n{traceback.format_exc()}"
        finally:
            sys.stdout = old_stdout
        return captured_output.getvalue()

    def gui_input_handler(self, prompt_message):
        self.master.attributes('-topmost', 1)
        self.master.attributes('-topmost', 0)
        self.master.focus_force()
        user_input = simpledialog.askstring("VM 输入请求", prompt_message, parent=self.master)
        return user_input

    def compile_and_run(self):
        source_code = self.source_code_text.get(1.0, tk.END).strip()
        if not source_code:
            self.status_text.config(state='normal')
            self.status_text.delete(1.0, tk.END)
            self.status_text.insert(tk.END, "错误: 源代码为空。")
            try:
                self.status_ans_notebook.select(0)
            except tk.TclError: pass
            return

        self._set_text_area_content(self.vm_code_text, "")
        self._set_text_area_content(self.symtable_text, "")
        self._set_text_area_content(self.ans_text, "")
        
        self.status_text.config(state='normal')
        self.status_text.delete(1.0, tk.END)
        self.status_text.insert(tk.END, "正在编译和运行...")
        
        try:
            self.status_ans_notebook.select(0)
        except tk.TclError: pass
        self.master.update_idletasks()

        generated_vm_code = None
        final_symtable = None
        vm_output = None

        try:
            my_lexer = Lexer(source_code)
            my_parser = ParserAndGenerator(my_lexer)
            generated_vm_code, final_symtable = my_parser.parse()

            vm_code_str_list = []
            for instr in generated_vm_code:
                l_str = str(instr['l']) if instr['l'] is not None else ""
                a_str = str(instr['a']) if instr['a'] is not None else ""
                nargs_str = f" nargs:{instr['nargs']}" if 'nargs' in instr else ""
                vm_code_str_list.append(
                    f"{instr['pc']:<3} {instr['op']:<5} {l_str:<5} {str(a_str):<12} {nargs_str:<10};{instr['comment']}"
                )
            self._set_text_area_content(self.vm_code_text, "\n".join(vm_code_str_list))

            symtable_display_str = self.capture_symbol_table_display(final_symtable)
            self._set_text_area_content(self.symtable_text, symtable_display_str)

            vm_instance = VM(generated_vm_code, debug_vm=False)
            vm_instance.set_input_callback(self.gui_input_handler)

            vm_output = vm_instance.run()

            ans_str_list = [f"ANS={item}" for item in vm_output] if vm_output else ["VM没有输出。"]
            self._set_text_area_content(self.ans_text, "\n".join(ans_str_list))
            try:
                self.status_ans_notebook.select(1)
            except tk.TclError: pass

            self.status_text.config(state='normal')
            self.status_text.delete(1.0, tk.END)
            self.status_text.insert(tk.END, "编译和执行成功。")

        except Exception as e:
            detailed_error = f"发生错误:\n{type(e).__name__}: {e}\n\nTraceback:\n{traceback.format_exc()}"
            self.status_text.config(state='normal')
            self.status_text.delete(1.0, tk.END)
            self.status_text.insert(tk.END, detailed_error)
            try:
                self.status_ans_notebook.select(0)
            except tk.TclError: pass

            if generated_vm_code is None:
                self._set_text_area_content(self.vm_code_text, "未能生成VM代码。")
            if final_symtable is None and generated_vm_code is not None :
                 self._set_text_area_content(self.symtable_text, "符号表生成可能存在问题或解析在此之前失败。")
            if vm_output is None and generated_vm_code is not None:
                 self._set_text_area_content(self.ans_text, "VM未运行或未能产生输出。")

def main_gui():
    root = tk.Tk()
    gui = CompilerGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main_gui()