# Get Certificate:
# Shell:
   
`echo | openssl s_client -connect esgf-node.ipsl.upmc.fr:443 -servername esgf-node.ipsl.upmc.fr 2>/dev/null | openssl x509 >> trusted-certificates.pem`
   
# Locate the CA Certificates:
# python:

`import certifi
print(certifi.where())`


# Add the Certificate to the `cacert.pem` file.
# Shell:

`cat trusted-certificates.pem >> {{path-to-your-cacert.pem}}`
   

# Using the Custom CA Bundle: without modifying the default `cacert.pem`
# python:

`import os
os.environ['REQUESTS_CA_BUNDLE'] = 'trusted-certificates.pem'`
   
