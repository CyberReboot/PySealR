# Goal of this fork
Using PySEAL library to implement a Machine Learning algorithm which can operate on homomorphically encrypted data.


This repository is a proof of concept for adding a Python wrapper to the
(Simple Encrypted Arithmetic Library (SEAL))[http://sealcrypto.org/], a homomorphic encryption library,
developed by researchers in the Cryptography Research Group at Microsoft Research. Currently, this uses
the version 2.3 codebase and dockerizes the library build (including a shared runtime), C++ example build,
and Python wrapper build. The Python wrapper is using pybind11.

To build the wrapped Python version of SEAL, first run the executable build-docker.sh. This creates a seal
package that can be imported in Python; to see examples of how to use this package in cryptography tasks,
run the executable run-docker.sh (which runs the examples implemented in SEALPythonExamples/examples.py).

When using the SEAL library for basic encryption tasks, the first step is to create a new EncryptionParameters
object, and to set its modulus attributes. The polynomial modulus should be set to a power-of-2 cyclotomic
polynomial (x^(2^n) + 1 for some n), the coefficient modulus should be set to seal.coeff_modulus_128(<int argument>)
(a good int argument is 2048), and the plain modulus can be set to any positive integer (using
a power of 2 or a prime would be a good choice).

After creating the EncryptionParameters object, we create a SEALContext object to store our parameters, passing in
our encryption parameters as the argument in the constructor for the context; the context, among other things, checks
the validity of the parameters.

The last objects to instantiate before we can complete various encryption tasks are:
- KeyGenerator object (which takes in the context as its constructor argument and generates a public/private key pair as
  well as evaluation keys).
- Encryptor object (which takes in the context and the public key as its constructor arguments and encrypts plaintext
  polynomials).
- Decryptor object (which takes in the context and the secret key as its constructor arguments and decrypts plaintext
  polynomials).
- IntegerEncoder object (which takes in the context's plain modulus as its constructor argument and encodes integers
  as plaintext polynomials and decodes plaintext polnomials as integers).
- Evaluator object (which takes in the context as its constructor argument, and performs computations on encrypted data
  and relinearizes encrypted data after multiplication operations).
- EvaluationKeys object (which requires no arguments for the constructor and is passed to a KeyGenerator to generate
  evaluation keys and is also passed to an Evaluator when relinearizing).

Once these objects are all instantiated, we can homomorphically encrypt and perform computations on encrypted data as follows:
- Use the IntegerEncoder object to encode our data as plaintext polynomials.
- Use the Encryptor object to encrypt the plaintext polynomials obtained above.
- Use the Evaluator object to perform computations on encrypted data (taking care to relinearize after multiplication operations).
- Use the Decryptor object to decrypt the encrypted computational output(s).
- Use the IntegerEncoder object to decode the decrypted output(s) (which are plaintext polynomials) and convert them back
  into integer data.
- Due to the homomorphic encryption scheme, the final data obtained with the above procedure should match the output of
  performing our computations on the raw, unencrypted data we started with, so long as we do not incur excessive corruption
  with our computations on the encrypted data (to check whether we are safe on this front, we can check to make sure that
  the noise budget of a given encrypted datum is sufficiently large; the consumption of this noise budget reflects the
  corruption incurred).

The above protocol is the simplest procedure and use case for SEAL homomorphic encryption, and can be seen in Example Basics: I
in our SEALPythonExamples/examples.py file. To see more sophisticated use cases, refer to the later examples in the same file.

Original home: https://www.microsoft.com/en-us/research/project/simple-encrypted-arithmetic-library/
