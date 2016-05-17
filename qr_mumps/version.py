version = '0.3.0'
release = False

if not release:
    s = version.split('.')
    version = s[0]+'.'+s[1]+'.'+'dev'+s[2]
