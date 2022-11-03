n = 20
levels = np.linspace(0.9, 0.0, n)

pws, fracs = [], []
for level in levels:
    pw, frac = energy_proj(f, level, ftr=ftr, normalize=True, return_frac=True)
    pws.append(pw)
    fracs.append(frac)
pws = pws / np.max(pws)


fig, ax = pplt.subplots()
ax.plot(
    levels[::-1], fracs[::-1], color='black',
    marker='.', lw=0,
)
ax.format(xlabel="Threshold (x-x'-y-y')", ylabel='Fraction of particles', xlim=(-0.02, 1.0))


fig, ax = pplt.subplots(figsize=(4, 1.75))
ax.pcolormesh(coords[4], levels[::-1], pws[::-1],
              colorbar=True, colorbar_kw=dict(label='Density (arb. units)', width=0.1))
ax.format(xlabel='Energy [MeV]', ylabel='4D thresh')
plt.show()


cmap = pplt.Colormap('fire_r', left=0.0, right=0.9)
# cmap = pplt.Colormap('crest', left=0.0, right=1.0)

fig, ax = pplt.subplots(figsize=(4, 1.75))
ax.plot(coords[4], pws[::-1].T, cycle=cmap, lw=1, colorbar=True, 
        colorbar_kw=dict(values=levels[::-1], label="Threshold (x-x'-y-y')"))
ax.format(xlabel="Energy [MeV]")
plt.show()



fig, ax = pplt.subplots(figsize=(4.5, 1.55))

alpha = 0.3
color = 'red6'
ax2 = ax.alty(color=color)
ax2.format(ylabel='Fraction of beam', 
           yscale='log', 
           ylim=(0.001, 1.0))

_levels = np.linspace(0.0, 0.95, 35)
_fracs = [energy_proj(f, _level, ftr=ftr, normalize=True, return_frac=True)[1]
          for _level in _levels]
ax2.plot(_levels[::-1], _fracs[::-1], zorder=0, color=color, alpha=alpha, lw=1.25)

for level, pw in zip(levels, pws):
    ax.plotx(coords[4], level + 0.045 * pw, 
             # color='black', alpha=0.3,
             color='black', alpha=1,
             zorder=999999)
ax.format(
    ylabel='Energy [MeV]', 
    xlabel="Threshold (x-x'-y-y')",
    ylim=(-0.09, 0.09), xlim=(-0.03, 0.97),
)
# plt.savefig('_output/waterfall')



# Z = pws[::-1]
# X, Y = np.meshgrid(levels[::-1], coords[4], indexing='ij')    
# lines = []
# line_marker = dict(color='black', width=3)
# for x, y, z in zip(X, Y, Z):
#     lines.append(go.Scatter3d(x=x, y=y, z=z, mode='lines', line=line_marker))
# uaxis= dict(
#     gridcolor='rgb(255, 255, 255)',
#     zerolinecolor='rgb(255, 255, 255)',
#     showbackground=True,
#     backgroundcolor='rgb(230, 230,230)',
# )
# layout = go.Layout(
#     width=500,
#     height=500,
#     showlegend=False,
#     scene=dict(
#         xaxis=uaxis, 
#         yaxis=uaxis,
#         zaxis=uaxis,
#     ),
# )
# fig = go.Figure(data=lines, layout=layout)
# fig.show()



# fig = go.Figure(data=[go.Surface(x=levels[::-1], y=coords[4], z=pws[::-1].T)])
# fig.update_layout(width=500, height=500)
# fig.show()