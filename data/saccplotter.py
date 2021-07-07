import tkinter as tk
import sacc
from tkinter import filedialog
import matplotlib.pyplot as plt
import numpy as np
import os


class SaccPlotter(object):
    def __init__(self):
        self.nfiles = 0
        self.saccs = {}
        self.main_w = tk.Tk()
        self.main_w.title("Sacc plotter")
        self.main_w.geometry('500x500')
        self.main_w.config(background='white')
        ll = tk.Label(self.main_w,
                      text='Sacc file explorer',
                      width=30, height=4, fg='blue')
        bexp = tk.Button(self.main_w,
                         text='Read file',
                         command=self.add_file)
        bnz = tk.Button(self.main_w,
                        text='Plot N(z)',
                        command=self.plot_nzs)
        bcl = tk.Button(self.main_w,
                        text='Plot Cls',
                        command=self.plot_cls)
        bexi = tk.Button(self.main_w,
                         text='Exit',
                         command=exit)
        self.lf = tk.Label(self.main_w,
                           text='Files loaded:\n', fg='blue',
                           width=30)

        ll.grid(column=1, row=1)
        bexp.grid(column=1, row=2)
        bnz.grid(column=1, row=3)
        bcl.grid(column=1, row=4)
        bexi.grid(column=1, row=5)
        self.lf.grid(column=1, row=6)

        self.main_w.mainloop()

    def _error_window(self, parent, msg, title):
        w = tk.Toplevel(parent)
        w.title(title)
        ll = tk.Label(w, text=msg)
        ll.pack()
        w.mainloop()

    def _get_dtype(self, s, t1, t2):
        s1 = s.tracers[t1].quantity
        s2 = s.tracers[t2].quantity
        if s1 == s2 == 'galaxy_density':
            return 'cl_00'
        if (((s1 == 'galaxy_density') and (s2 == 'galaxy_shear')) or
                ((s2 == 'galaxy_shear') and (s2 == 'galaxy_density'))):
            return 'cl_0e'
        if s1 == s2 == 'galaxy_shear':
            return 'cl_ee'

    def plot_nzs(self):
        nzlist = {}
        w = tk.Toplevel(self.main_w)
        w.title('Plot N(z)')
        w.geometry('500x500')
        tkvar = tk.StringVar(w)
        tkvar.set('')
        menu = tk.OptionMenu(w, tkvar, *(self.saccs.keys()))
        ll = tk.Label(w,
                      text='N(z)s to plot:\n', fg='blue')

        def select_tracer(*args):
            fname = tkvar.get()
            if fname not in nzlist:
                nzlist[fname] = []
            s = self.saccs[fname]['sacc']
            ww = tk.Toplevel(w)
            ww.title("Select tracer combination")
            ww.geometry('700x300')
            tnames = list(s.tracers.keys())
            v1 = tk.StringVar(ww)
            v1.set(tnames[0])
            m1 = tk.OptionMenu(ww, v1, *tnames)
            m1.grid(column=1, row=1)

            def add_nz():
                t1 = v1.get()
                if s.tracers[t1].tracer_type == 'NZ':
                    nzlist[fname].append(t1)
                    ww.destroy()
                    ll.config(text=ll.cget('text')+f'{fname} {t1}\n')
                else:
                    self._error_window(ww, f"{t1} is not an NZ tracer",
                                       "Wrong tracer")

            b = tk.Button(ww, text='Add N(z)',
                          command=add_nz)
            b.grid(column=1, row=2)

        def plotting():
            for fname, trs in nzlist.items():
                s = self.saccs[fname]['sacc']
                for t1 in trs:
                    t = s.tracers[t1]
                    plt.plot(t.z, t.nz, '-', label=f'{fname}, {t1}')
            plt.legend()
            plt.show()

        tkvar.trace('w', select_tracer)
        bplot = tk.Button(w,
                          text='Plot',
                          command=plotting)
        menu.grid(column=1, row=1)
        bplot.grid(column=1, row=2)
        ll.grid(column=1, row=3)
        w.mainloop()

    def plot_cls(self):
        cllist = {}
        w = tk.Toplevel(self.main_w)
        w.title('Plot Cls')
        w.geometry('500x500')
        tkvar = tk.StringVar(w)
        tkvar.set('')
        menu = tk.OptionMenu(w, tkvar, *(self.saccs.keys()))
        ll = tk.Label(w,
                      text='Cls to plot:\n', fg='blue')

        def select_tracer_combo(*args):
            fname = tkvar.get()
            if fname not in cllist:
                cllist[fname] = []
            s = self.saccs[fname]['sacc']
            ww = tk.Toplevel(w)
            ww.title("Select tracer combination")
            ww.geometry('700x300')
            tnames = list(s.tracers.keys())
            v1 = tk.StringVar(ww)
            v1.set(tnames[0])
            v2 = tk.StringVar(ww)
            v2.set(tnames[0])
            m1 = tk.OptionMenu(ww, v1, *tnames)
            m2 = tk.OptionMenu(ww, v2, *tnames)
            m1.grid(column=1, row=1)
            m2.grid(column=2, row=1)

            def add_combo():
                t1 = v1.get()
                t2 = v2.get()
                if (t1, t2) in self.saccs[fname]['combos']:
                    cllist[fname].append((t1, t2))
                    ww.destroy()
                    ll.config(text=ll.cget('text')+f'{fname} {t1} x {t2}\n')
                else:
                    self._error_window(ww, f"{t1} x {t2} not found",
                                       "Cl not found")

            b = tk.Button(ww, text='Add combination',
                          command=add_combo)
            b.grid(column=1, row=2)

        def plotting():
            for fname, combos in cllist.items():
                s = self.saccs[fname]['sacc']
                for t1, t2 in combos:
                    l, cl, cov = s.get_ell_cl(self._get_dtype(s, t1, t2),
                                              t1, t2, return_cov=True)
                    plt.errorbar(l, cl, yerr=np.sqrt(np.diag(cov)),
                                 fmt='.', label=f'{fname}, {t1}x{t2}')
            plt.loglog()
            plt.legend()
            plt.show()

        tkvar.trace('w', select_tracer_combo)
        bplot = tk.Button(w,
                          text='Plot',
                          command=plotting)
        menu.grid(column=1, row=1)
        bplot.grid(column=1, row=2)
        ll.grid(column=1, row=3)
        w.mainloop()

    def add_file(self):
        fname = filedialog.askopenfilename(initialdir='./',
                                           title='Select Sacc file',
                                           filetypes=(('Fits files',
                                                       '*.fits*'),
                                                      ('all files',
                                                       '*.*')))
        try:
            s = sacc.Sacc.load_fits(fname)
        except OSError:
            self._error_window(self.main_w,
                               f"Couldn't open file \n {fname}",
                               "File open error")
            return
        self.nfiles += 1
        self.saccs['File %d' % (self.nfiles)] = {
            'sacc': s,
            'combos': s.get_tracer_combinations()}
        self.lf.config(text=self.lf.cget('text') +
                       f'File {self.nfiles}, {os.path.split(fname)[-1]}\n')


sp = SaccPlotter()
