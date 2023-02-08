import iadpython as iad

mu_s = 10  # scattering coefficient [1/mm]
mu_a = 0.1 # absorption coefficient [1/mm]
g = 0.9    # scattering anisotropy
d = 1      # thickness mm

a = mu_s/(mu_a+mu_s)
b = mu_s/(mu_a+mu_s) * d

# air / glass / sample / glass / air
s = iad.Sample(a=a, b=b, g=g, n=1.4, n_above=1.5, n_below=1.5)
ur1, ut1, uru, utu = s.rt()

print('Collimated light incident perpendicular to sample')
print('  total reflection = %.5f' % ur1)
print('  total transmission = %.5f' % ut1)

print('Diffuse light incident on sample')
print('  total reflection = %.5f' % uru)
print('  total transmission = %.5f' % utu)