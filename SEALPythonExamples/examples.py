import time
import random
import pickle
import threading
import numpy as np
import pandas as pd
import seal
from seal import ChooserEvaluator, \
	Ciphertext, \
	Decryptor, \
	Encryptor, \
	EncryptionParameters, \
	Evaluator, \
	IntegerEncoder, \
	FractionalEncoder, \
	KeyGenerator, \
	MemoryPoolHandle, \
	Plaintext, \
	SEALContext, \
	EvaluationKeys, \
	GaloisKeys, \
	PolyCRTBuilder, \
	ChooserEncoder, \
	ChooserEvaluator, \
	ChooserPoly


def example_basics_i():
	print_example_banner("Example: Basics I");

	# In this example we demonstrate setting up encryption parameters and other
	# relevant objects for performing simple computations on encrypted integers.

	# SEAL uses the Fan-Vercauteren (FV) homomorphic encryption scheme. We refer to
	# https://eprint.iacr.org/2012/144 for full details on how the FV scheme works.
	# For better performance, SEAL implements the "FullRNS" optimization of FV, as
	# described in https://eprint.iacr.org/2016/510.

	# The first task is to set up an instance of the EncryptionParameters class.
	# It is critical to understand how these different parameters behave, how they
	# affect the encryption scheme, performance, and the security level. There are
	# three encryption parameters that are necessary to set:

	#     - poly_modulus (polynomial modulus);
	#     - coeff_modulus ([ciphertext] coefficient modulus);
	#     - plain_modulus (plaintext modulus).

	# A fourth parameter -- noise_standard_deviation -- has a default value of 3.19
	# and should not be necessary to modify unless the user has a specific reason
	# to and knows what they are doing.

	# The encryption scheme implemented in SEAL cannot perform arbitrary computations
	# on encrypted data. Instead, each ciphertext has a specific quantity called the
	# `invariant noise budget' -- or `noise budget' for short -- measured in bits.
	# The noise budget of a freshly encrypted ciphertext (initial noise budget) is
	# determined by the encryption parameters. Homomorphic operations consume the
	# noise budget at a rate also determined by the encryption parameters. In SEAL
	# the two basic homomorphic operations are additions and multiplications, of
	# which additions can generally be thought of as being nearly free in terms of
	# noise budget consumption compared to multiplications. Since noise budget
	# consumption is compounding in sequential multiplications, the most significant
	# factor in choosing appropriate encryption parameters is the multiplicative
	# depth of the arithmetic circuit that needs to be evaluated. Once the noise
	# budget in a ciphertext reaches zero, it becomes too corrupted to be decrypted.
	# Thus, it is essential to choose the parameters to be large enough to support
	# the desired computation; otherwise the result is impossible to make sense of
	# even with the secret key.
	parms = EncryptionParameters()

	# We first set the polynomial modulus. This must be a power-of-2 cyclotomic
	# polynomial, i.e. a polynomial of the form "1x^(power-of-2) + 1". The polynomial
	# modulus should be thought of mainly affecting the security level of the scheme;
	# larger polynomial modulus makes the scheme more secure. At the same time, it
	# makes ciphertext sizes larger, and consequently all operations slower.
	# Recommended degrees for poly_modulus are 1024, 2048, 4096, 8192, 16384, 32768,
	# but it is also possible to go beyond this. Since we perform only a very small
	# computation in this example, it suffices to use a very small polynomial modulus
	parms.set_poly_modulus("1x^2048 + 1")

	# Next we choose the [ciphertext] coefficient modulus (coeff_modulus). The size
	# of the coefficient modulus should be thought of as the most significant factor
	# in determining the noise budget in a freshly encrypted ciphertext: bigger means
	# more noise budget. Unfortunately, a larger coefficient modulus also lowers the
	# security level of the scheme. Thus, if a large noise budget is required for
	# complicated computations, a large coefficient modulus needs to be used, and the
	# reduction in the security level must be countered by simultaneously increasing
	# the polynomial modulus.

	# To make parameter selection easier for the user, we have constructed sets of
	# largest allowed coefficient moduli for 128-bit and 192-bit security levels
	# for different choices of the polynomial modulus. These recommended parameters
	# follow the Security white paper at http://HomomorphicEncryption.org. However,
	# due to the complexity of this topic, we highly recommend the user to directly
	# consult an expert in homomorphic encryption and RLWE-based encryption schemes
	# to determine the security of their parameter choices.

	# Our recommended values for the coefficient modulus can be easily accessed
	# through the functions

	#     coeff_modulus_128bit(int)
	#     coeff_modulus_192bit(int)

	# for 128-bit and 192-bit security levels. The integer parameter is the degree
	# of the polynomial modulus.

	# In SEAL the coefficient modulus is a positive composite number -- a product
	# of distinct primes of size up to 60 bits. When we talk about the size of the
	# coefficient modulus we mean the bit length of the product of the small primes.
	# The small primes are represented by instances of the SmallModulus class; for
	# example coeff_modulus_128bit(int) returns a vector of SmallModulus instances.

	# It is possible for the user to select their own small primes. Since SEAL uses
	# the Number Theoretic Transform (NTT) for polynomial multiplications modulo the
	# factors of the coefficient modulus, the factors need to be prime numbers
	# congruent to 1 modulo 2*degree(poly_modulus). We have generated a list of such
	# prime numbers of various sizes, that the user can easily access through the
	# functions

	#     small_mods_60bit(int)
	#     small_mods_50bit(int)
	#     small_mods_40bit(int)
	#     small_mods_30bit(int)

	# each of which gives access to an array of primes of the denoted size. These
	# primes are located in the source file util/globals.cpp.

	# Performance is mainly affected by the size of the polynomial modulus, and the
	# number of prime factors in the coefficient modulus. Thus, it is important to
	# use as few factors in the coefficient modulus as possible.

	# In this example we use the default coefficient modulus for a 128-bit security
	# level. Concretely, this coefficient modulus consists of only one 56-bit prime
	# factor: 0xfffffffff00001.
	parms.set_coeff_modulus(seal.coeff_modulus_128(2048))

	# The plaintext modulus can be any positive integer, even though here we take
	# it to be a power of two. In fact, in many cases one might instead want it to
	# be a prime number; we will see this in example_batching(). The plaintext
	# modulus determines the size of the plaintext data type, but it also affects
	# the noise budget in a freshly encrypted ciphertext, and the consumption of
	# the noise budget in homomorphic multiplication. Thus, it is essential to try
	# to keep the plaintext data type as small as possible for good performance.
	# The noise budget in a freshly encrypted ciphertext is

	#     ~ log2(coeff_modulus/plain_modulus) (bits)

	# and the noise budget consumption in a homomorphic multiplication is of the
	# form log2(plain_modulus) + (other terms).
	parms.set_plain_modulus(1 << 8)

	# Now that all parameters are set, we are ready to construct a SEALContext
	# object. This is a heavy class that checks the validity and properties of
	# the parameters we just set, and performs and stores several important
	# pre-computations.
	context = SEALContext(parms)

	# Print the parameters that we have chosen
	print_parameters(context);

	# Plaintexts in the FV scheme are polynomials with coefficients integers modulo
	# plain_modulus. To encrypt for example integers instead, one can use an
	# `encoding scheme' to represent the integers as such polynomials. SEAL comes
	# with a few basic encoders:

	# [IntegerEncoder]
	# Given an integer base b, encodes integers as plaintext polynomials as follows.
	# First, a base-b expansion of the integer is computed. This expansion uses
	# a `balanced' set of representatives of integers modulo b as the coefficients.
	# Namely, when b is odd the coefficients are integers between -(b-1)/2 and
	# (b-1)/2. When b is even, the integers are between -b/2 and (b-1)/2, except
	# when b is two and the usual binary expansion is used (coefficients 0 and 1).
	# Decoding amounts to evaluating the polynomial at x=b. For example, if b=2,
	# the integer

	#     26 = 2^4 + 2^3 + 2^1

	# is encoded as the polynomial 1x^4 + 1x^3 + 1x^1. When b=3,

	#     26 = 3^3 - 3^0

	# is encoded as the polynomial 1x^3 - 1. In memory polynomial coefficients are
	# always stored as unsigned integers by storing their smallest non-negative
	# representatives modulo plain_modulus. To create a base-b integer encoder,
	# use the constructor IntegerEncoder(plain_modulus, b). If no b is given, b=2
	# is used.

	# [FractionalEncoder]
	# The FractionalEncoder encodes fixed-precision rational numbers as follows.
	# It expands the number in a given base b, possibly truncating an infinite
	# fractional part to finite precision, e.g.

	#     26.75 = 2^4 + 2^3 + 2^1 + 2^(-1) + 2^(-2)

	# when b=2. For the sake of the example, suppose poly_modulus is 1x^1024 + 1.
	# It then represents the integer part of the number in the same way as in
	# IntegerEncoder (with b=2 here), and moves the fractional part instead to the
	# highest degree part of the polynomial, but with signs of the coefficients
	# changed. In this example we would represent 26.75 as the polynomial

	#     -1x^1023 - 1x^1022 + 1x^4 + 1x^3 + 1x^1.

	# In memory the negative coefficients of the polynomial will be represented as
	# their negatives modulo plain_modulus.

	# [PolyCRTBuilder]
	# If plain_modulus is a prime congruent to 1 modulo 2*degree(poly_modulus), the
	# plaintext elements can be viewed as 2-by-(degree(poly_modulus) / 2) matrices
	# with elements integers modulo plain_modulus. When a desired computation can be
	# vectorized, using PolyCRTBuilder can result in massive performance improvements
	# over naively encrypting and operating on each input number separately. Thus,
	# in more complicated computations this is likely to be by far the most important
	# and useful encoder. In example_batching() we show how to use and operate on
	# encrypted matrix plaintexts.

	# For performance reasons, in homomorphic encryption one typically wants to keep
	# the plaintext data types as small as possible, which can make it challenging to
	# prevent data type overflow in more complicated computations, especially when
	# operating on rational numbers that have been scaled to integers. When using
	# PolyCRTBuilder estimating whether an overflow occurs is a fairly standard task,
	# as the matrix slots are integers modulo plain_modulus, and each slot is operated
	# on independently of the others. When using IntegerEncoder or FractionalEncoder
	# it is substantially more difficult to estimate when an overflow occurs in the
	# plaintext, and choosing the plaintext modulus very carefully to be large enough
	# is critical to avoid unexpected results. Specifically, one needs to estimate how
	# large the largest coefficient in  the polynomial view of all of the plaintext
	# elements becomes, and choose the plaintext modulus to be larger than this value.
	# SEAL comes with an automatic parameter selection tool that can help with this
	# task, as is demonstrated in example_parameter_selection().

	# Here we choose to create an IntegerEncoder with base b=2.
	encoder = IntegerEncoder(context.plain_modulus())

	# We are now ready to generate the secret and public keys. For this purpose we need
	# an instance of the KeyGenerator class. Constructing a KeyGenerator automatically
	# generates the public and secret key, which can then be read to local variables.
	# To create a fresh pair of keys one can call KeyGenerator::generate() at any time.
	keygen = KeyGenerator(context)
	public_key = keygen.public_key()
	secret_key = keygen.secret_key()

	# To be able to encrypt, we need to construct an instance of Encryptor. Note that
	# the Encryptor only requires the public key.
	encryptor = Encryptor(context, public_key)

	# Computations on the ciphertexts are performed with the Evaluator class.
	evaluator = Evaluator(context)

	# We will of course want to decrypt our results to verify that everything worked,
	# so we need to also construct an instance of Decryptor. Note that the Decryptor
	# requires the secret key.
	decryptor = Decryptor(context, secret_key)

	# We start by encoding two integers as plaintext polynomials.
	value1 = 55;
	plain1 = encoder.encode(value1);
	print("Encoded " + (str)(value1) + " as polynomial " + plain1.to_string() + " (plain1)")

	value2 = -79;
	plain2 = encoder.encode(value2);
	print("Encoded " + (str)(value2) + " as polynomial " + plain2.to_string() + " (plain2)")

	# Encrypting the values is easy.
	encrypted1 = Ciphertext()
	encrypted2 = Ciphertext()
	print("Encrypting plain1: ")
	encryptor.encrypt(plain1, encrypted1)
	print("Done (encrypted1)")

	print("Encrypting plain2: ")
	encryptor.encrypt(plain2, encrypted2)
	print("Done (encrypted2)")

	# To illustrate the concept of noise budget, we print the budgets in the fresh
	# encryptions.
	print("Noise budget in encrypted1: " + (str)(decryptor.invariant_noise_budget(encrypted1)) + " bits")
	print("Noise budget in encrypted2: " + (str)(decryptor.invariant_noise_budget(encrypted2)) + " bits")

	# As a simple example, we compute (-encrypted1 + encrypted2) * encrypted2.

	# Negation is a unary operation.
	evaluator.negate(encrypted1)

	# Negation does not consume any noise budget.
	print("Noise budget in -encrypted1: " + (str)(decryptor.invariant_noise_budget(encrypted1)) + " bits")

	# Addition can be done in-place (overwriting the first argument with the result,
	# or alternatively a three-argument overload with a separate destination variable
	# can be used. The in-place variants are always more efficient. Here we overwrite
	# encrypted1 with the sum.
	evaluator.add(encrypted1, encrypted2)

	# It is instructive to think that addition sets the noise budget to the minimum
	# of the input noise budgets. In this case both inputs had roughly the same
	# budget going on, and the output (in encrypted1) has just slightly lower budget.
	# Depending on probabilistic effects, the noise growth consumption may or may
	# not be visible when measured in whole bits.
	print("Noise budget in -encrypted1 + encrypted2: " + (str)(decryptor.invariant_noise_budget(encrypted1)) + " bits")

	# Finally multiply with encrypted2. Again, we use the in-place version of the
	# function, overwriting encrypted1 with the product.
	evaluator.multiply(encrypted1, encrypted2)

	# Multiplication consumes a lot of noise budget. This is clearly seen in the
	# print-out. The user can change the plain_modulus to see its effect on the
	# rate of noise budget consumption.
	print("Noise budget in (-encrypted1 + encrypted2) * encrypted2: " + (str)(decryptor.invariant_noise_budget(encrypted1)) + " bits")

	# Now we decrypt and decode our result.
	plain_result = Plaintext()
	print("Decrypting result: ")
	decryptor.decrypt(encrypted1, plain_result)
	print("Done")

	# Print the result plaintext polynomial.
	print("Plaintext polynomial: " + plain_result.to_string())

	# Decode to obtain an integer result.
	print("Decoded integer: " + (str)(encoder.decode_int32(plain_result)))


def example_basics_ii():
	print_example_banner("Example: Basics II")

	# In this example we explain what relinearization is, how to use it, and how
	# it affects noise budget consumption.

	# First we set the parameters, create a SEALContext, and generate the public
	# and secret keys. We use slightly larger parameters than be fore to be able
	# to do more homomorphic multiplications.
	parms = EncryptionParameters()
	parms.set_poly_modulus("1x^8192 + 1")

	# The default coefficient modulus consists of the following primes:

	#     0x7fffffffba0001,
	#     0x7fffffffaa0001,
	#     0x7fffffff7e0001,
	#     0x3fffffffd60001.

	# The total size is 219 bits.
	parms.set_coeff_modulus(seal.coeff_modulus_128(8192))
	parms.set_plain_modulus(1 << 10)

	context = SEALContext(parms)
	print_parameters(context)

	keygen = KeyGenerator(context)
	public_key = keygen.public_key()
	secret_key = keygen.secret_key()

	# We also set up an Encryptor, Evaluator, and Decryptor here. We will
	# encrypt polynomials directly in this example, so there is no need for
	# an encoder.
	encryptor = Encryptor(context, public_key)
	evaluator = Evaluator(context)
	decryptor = Decryptor(context, secret_key)

	# There are actually two more types of keys in SEAL: `evaluation keys' and
	# `Galois keys'. Here we will discuss evaluation keys, and Galois keys will
	# be discussed later in example_batching().

	# In SEAL, a valid ciphertext consists of two or more polynomials with
	# coefficients integers modulo the product of the primes in coeff_modulus.
	# The current size of a ciphertext can be found using Ciphertext::size().
	# A freshly encrypted ciphertext always has size 2.
	#plain1 = Plaintext("1x^2 + 2x^1 + 3")
	plain1 = Plaintext("1x^2 + 2x^1 + 3")
	encrypted = Ciphertext()

	print("")
	print("Encrypting " + plain1.to_string() + ": ")
	encryptor.encrypt(plain1, encrypted)
	print("Done")
	print("Size of a fresh encryption: " + (str)(encrypted.size()))
	print("Noise budget in fresh encryption: " + (str)(decryptor.invariant_noise_budget(encrypted)) + " bits")

	# Homomorphic multiplication results in the output ciphertext growing in size.
	# More precisely, if the input ciphertexts have size M and N, then the output
	# ciphertext after homomorphic multiplication will have size M+N-1. In this
	# case we square encrypted twice to observe this growth (also observe noise
	# budget consumption).
	evaluator.square(encrypted)
	print("Size after squaring: " + (str)(encrypted.size()))
	print("Noise budget after squaring: " + (str)(decryptor.invariant_noise_budget(encrypted)) + " bits")
	plain2 = Plaintext()
	decryptor.decrypt(encrypted, plain2)
	print("Second power: " + plain2.to_string())

	evaluator.square(encrypted);
	print("Size after squaring: " + (str)(encrypted.size()))
	print("Noise budget after squaring: " + (str)(decryptor.invariant_noise_budget(encrypted)) + " bits")

	# It does not matter that the size has grown -- decryption works as usual.
	# Observe from the print-out that the coefficients in the plaintext have
	# grown quite large. One more squaring would cause some of them to wrap
	# around plain_modulus (0x400), and as a result we would no longer obtain
	# the expected result as an integer-coefficient polynomial. We can fix this
	# problem to some extent by increasing plain_modulus. This would make sense,
	# since we still have plenty of noise budget left.
	plain2 = Plaintext()
	decryptor.decrypt(encrypted, plain2)
	print("Fourth power: " + plain2.to_string())

	# The problem here is that homomorphic operations on large ciphertexts are
	# computationally much more costly than on small ciphertexts. Specifically,
	# homomorphic multiplication on input ciphertexts of size M and N will require
	# O(M*N) polynomial multiplications to be performed, and an addition will
	# require O(M+N) additions. Relinearization reduces the size of the ciphertexts
	# after multiplication back to the initial size (2). Thus, relinearizing one
	# or both inputs before the next multiplication, or e.g. before serializing the
	# ciphertexts, can have a huge positive impact on performance.

	# Another problem is that the noise budget consumption in multiplication is
	# bigger when the input ciphertexts sizes are bigger. In a complicated
	# computation the contribution of the sizes to the noise budget consumption
	# can actually become the dominant term. We will point this out again below
	# once we get to our example.

	# Relinearization itself has both a computational cost and a noise budget cost.
	# These both depend on a parameter called `decomposition bit count', which can
	# be any integer at least 1 [dbc_min()] and at most 60 [dbc_max()]. A large
	# decomposition bit count makes relinearization fast, but consumes more noise
	# budget. A small decomposition bit count can make relinearization slower, but
	# might not change the noise budget by any observable amount.

	# Relinearization requires a special type of key called `evaluation keys'.
	# These can be created by the KeyGenerator for any decomposition bit count.
	# To relinearize a ciphertext of size M >= 2 back to size 2, we actually need
	# M-2 evaluation keys. Attempting to relinearize a too large ciphertext with
	# too few evaluation keys will result in an exception being thrown.

	# We repeat our computation, but this time relinearize after both squarings.
	# Since our ciphertext never grows past size 3 (we relinearize after every
	# multiplication), it suffices to generate only one evaluation key.

	# First, we need to create evaluation keys. We use a decomposition bit count
	# of 16 here, which can be thought of as quite small.
	ev_keys16 = EvaluationKeys()

	# This function generates one single evaluation key. Another overload takes
	# the number of keys to be generated as an argument, but one is all we need
	# in this example (see above).
	keygen.generate_evaluation_keys(16, ev_keys16)

	print("")
	print("Encrypting " + plain1.to_string() + ": ")
	encryptor.encrypt(plain1, encrypted)
	print("Done")
	print("Size of a fresh encryption: " + (str)(encrypted.size()))
	print("Noise budget in fresh encryption: " + (str)(decryptor.invariant_noise_budget(encrypted)) + " bits")

	evaluator.square(encrypted)
	print("Size after squaring: " + (str)(encrypted.size()))
	print("Noise budget after squaring: " + (str)(decryptor.invariant_noise_budget(encrypted)) + " bits")

	evaluator.relinearize(encrypted, ev_keys16)
	print("Size after relinearization: " + (str)(encrypted.size()))
	print("Noise budget after relinearizing (dbs = " + (str)(ev_keys16.decomposition_bit_count()) + "): " +
	      (str)(decryptor.invariant_noise_budget(encrypted)) + " bits")

	evaluator.square(encrypted)
	print("Size after second squaring: " + (str)(encrypted.size()) + " bits")
	print("Noise budget after second squaring: " + (str)(decryptor.invariant_noise_budget(encrypted)) + " bits")

	evaluator.relinearize(encrypted, ev_keys16)
	print("Size after relinearization: " + (str)(encrypted.size()))
	print("Noise budget after relinearizing (dbs = " + (str)(ev_keys16.decomposition_bit_count()) + "): " +
	      (str)(decryptor.invariant_noise_budget(encrypted)) + " bits")

	decryptor.decrypt(encrypted, plain2)
	print("Fourth power: " + plain2.to_string())

	# Of course the result is still the same, but this time we actually
	# used less of our noise budget. This is not surprising for two reasons:

	#     - We used a very small decomposition bit count, which is why
	#       relinearization itself did not consume the noise budget by any
	#       observable amount;
	#     - Since our ciphertext sizes remain small throughout the two
	#       squarings, the noise budget consumption rate in multiplication
	#       remains as small as possible. Recall from above that operations
	#       on larger ciphertexts actually cause more noise growth.

	# To make matters even more clear, we repeat the computation a third time,
	# now using the largest possible decomposition bit count (60). We are not
	# measuring the time here, but relinearization with these evaluation keys
	# is significantly faster than with ev_keys16.
	ev_keys60 = EvaluationKeys()
	keygen.generate_evaluation_keys(seal.dbc_max(), ev_keys60)

	print("")
	print("Encrypting " + plain1.to_string() + ": ")
	encryptor.encrypt(plain1, encrypted)
	print("Done")
	print("Size of a fresh encryption: " + (str)(encrypted.size()))
	print("Noise budget in fresh encryption: " + (str)(decryptor.invariant_noise_budget(encrypted)) + " bits")

	evaluator.square(encrypted)
	print("Size after squaring: " + (str)(encrypted.size()))
	print("Noise budget after squaring: " + (str)(decryptor.invariant_noise_budget(encrypted)) + " bits")
	evaluator.relinearize(encrypted, ev_keys60)
	print("Size after relinearization: " + (str)(encrypted.size()))
	print("Noise budget after relinearizing (dbc = " + (str)(ev_keys60.decomposition_bit_count()) +
	      "): " + (str)(decryptor.invariant_noise_budget(encrypted)) + " bits")

	evaluator.square(encrypted)
	print("Size after second squaring: " + (str)(encrypted.size()))
	print("Noise budget after second squaring: " + (str)(decryptor.invariant_noise_budget(encrypted)) + " bits")
	evaluator.relinearize(encrypted, ev_keys60)
	print("Size after relinearization: " + (str)(encrypted.size()))
	print("Noise budget after relinearizing (dbc = " + (str)(ev_keys60.decomposition_bit_count()) +
	      "): " + (str)(decryptor.invariant_noise_budget(encrypted)) + " bits")

	decryptor.decrypt(encrypted, plain2)
	print("Fourth power: " + plain2.to_string())

	# Observe from the print-out that we have now used significantly more of our
	# noise budget than in the two previous runs. This is again not surprising,
	# since the first relinearization chops off a huge part of the noise budget.

	# However, note that the second relinearization does not change the noise
	# budget by any observable amount. This is very important to understand when
	# optimal performance is desired: relinearization always drops the noise
	# budget from the maximum (freshly encrypted ciphertext) down to a fixed
	# amount depending on the encryption parameters and the decomposition bit
	# count. On the other hand, homomorphic multiplication always consumes the
	# noise budget from its current level. This is why the second relinearization
	# does not change the noise budget anymore: it is already consumed past the
	# fixed amount determinted by the decomposition bit count and the encryption
	# parameters.

	# We now perform a third squaring and observe an even further compounded
	# decrease in the noise budget. Again, relinearization does not consume the
	# noise budget at this point by any observable amount, even with the largest
	# possible decomposition bit count.
	evaluator.square(encrypted)
	print("Size after third squaring " + (str)(encrypted.size()))
	print("Noise budget after third squaring: " + (str)(decryptor.invariant_noise_budget(encrypted)) + " bits")
	evaluator.relinearize(encrypted, ev_keys60)
	print("Size after relinearization: " + (str)(encrypted.size()))
	print("Noise budget after relinearizing (dbc = " + (str)(ev_keys60.decomposition_bit_count()) +
	      "): " + (str)(decryptor.invariant_noise_budget(encrypted)) + " bits")

	decryptor.decrypt(encrypted, plain2)
	print("Eighth power: " + plain2.to_string())

	# Observe from the print-out that the polynomial coefficients are no longer
	# correct as integers: they have been reduced modulo plain_modulus, and there
	# was no warning sign about this. It might be necessary to carefully analyze
	# the computation to make sure such overflow does not occur unexpectedly.

	# These experiments suggest that an optimal strategy might be to relinearize
	# first with evaluation keys with a small decomposition bit count, and later
	# with evaluation keys with a larger decomposition bit count (for performance)
	# when noise budget has already been consumed past the bound determined by the
	# larger decomposition bit count. For example, above the best strategy might
	# have been to use ev_keys16 in the first relinearization, and ev_keys60 in the
	# next two relinearizations for optimal noise budget consumption/performance
	# trade-off. Luckily, in most use-cases it is not so critical to squeeze out
	# every last bit of performance, especially when slightly larger parameters
	# are used.


def example_pickle():
	print_example_banner("Example: Ability to Pickle SEAL Ciphertexts")

	# In this example we demonstrate how to serialize SEAL Ciphertexts to files using Pickle.
	parms = EncryptionParameters()
	parms.set_poly_modulus("1x^2048 + 1")
	parms.set_coeff_modulus(seal.coeff_modulus_128(2048))
	parms.set_plain_modulus(1 << 8)

	context = SEALContext(parms)
	keygen = KeyGenerator(context)
	public_key = keygen.public_key()
	secret_key = keygen.secret_key()

	encryptor = Encryptor(context, public_key)
	decryptor = Decryptor(context, secret_key)

	# Create and encrypt a plaintext
	plain1 = Plaintext("1x^2 + 2x^1 + 3")

	print("")
	print("Encrypting plaintext '{}'".format(plain1.to_string()))
	encrypted1 = Ciphertext()
	encryptor.encrypt(plain1, encrypted1)

	# Serialize it using Pickle
	filename1 = "encrypted1.txt"
	print("Dumping plaintext '{}' to file '{}'".format(plain1.to_string(), filename1))
	pickle.dump(encrypted1, open(filename1, "wb"))

	# Read serialized ciphertext back in
	pickle_encrypted = pickle.load(open(filename1, "rb"))

	decrypted = Plaintext()
	decryptor.decrypt(pickle_encrypted, decrypted)
	print("Read serialized ciphertext back in and decrypt to: '{}'".format(decrypted.to_string()))


def example_weighted_average():
	print_example_banner("Example: Weighted Average")

	# In this example we demonstrate the FractionalEncoder, and use it to compute
	# a weighted average of 10 encrypted rational numbers. In this computation we
	# perform homomorphic multiplications of ciphertexts by plaintexts, which is
	# much faster than regular multiplications of ciphertexts by ciphertexts.
	# Moreover, such `plain multiplications' never increase the ciphertext size,
	# which is why we have no need for evaluation keys in this example.

	# We start by creating encryption parameters, setting up the SEALContext, keys,
	# and other relevant objects. Since our computation has multiplicative depth of
	# only two, it suffices to use a small poly_modulus.
	parms = EncryptionParameters()
	parms.set_poly_modulus("1x^2048 + 1")
	parms.set_coeff_modulus(seal.coeff_modulus_128(2048))
	parms.set_plain_modulus(1 << 8)

	context = SEALContext(parms)
	print_parameters(context)

	keygen = KeyGenerator(context)
	public_key = keygen.public_key()
	secret_key = keygen.secret_key()

	# We also set up an Encryptor, Evaluator, and Decryptor here.
	encryptor = Encryptor(context, public_key)
	evaluator = Evaluator(context)
	decryptor = Decryptor(context, secret_key)

	# Create a vector of 10 rational numbers (as doubles).
	rational_numbers = [3.1, 4.159, 2.65, 3.5897, 9.3, 2.3, 8.46, 2.64, 3.383, 2.795]

	# Create a vector of weights.
	coefficients = [0.1, 0.05, 0.05, 0.2, 0.05, 0.3, 0.1, 0.025, 0.075, 0.05]

	# We need a FractionalEncoder to encode the rational numbers into plaintext
	# polynomials. In this case we decide to reserve 64 coefficients of the
	# polynomial for the integral part (low-degree terms) and expand the fractional
	# part to 32 digits of precision (in base 3) (high-degree terms). These numbers
	# can be changed according to the precision that is needed; note that these
	# choices leave a lot of unused space in the 2048-coefficient polynomials.
	encoder = FractionalEncoder(context.plain_modulus(), context.poly_modulus(), 64, 32, 3)

	# We create a vector of ciphertexts for encrypting the rational numbers.
	encrypted_rationals = []
	rational_numbers_string = "Encoding and encrypting: "
	for i in range(10):
		# We create our Ciphertext objects into the vector by passing the
		# encryption parameters as an argument to the constructor. This ensures
		# that enough memory is allocated for a size 2 ciphertext. In this example
		# our ciphertexts never grow in size (plain multiplication does not cause
		# ciphertext growth), so we can expect the ciphertexts to remain in the same
		# location in memory throughout the computation. In more complicated examples
		# one might want to call a constructor that reserves enough memory for the
		# ciphertext to grow to a specified size to avoid costly memory moves when
		# multiplications and relinearizations are performed.
		encrypted_rationals.append(Ciphertext(parms))
		encryptor.encrypt(encoder.encode(rational_numbers[i]), encrypted_rationals[i])
		rational_numbers_string += (str)(rational_numbers[i])[:6]
		if i < 9: rational_numbers_string += ", "
	print(rational_numbers_string)

	# Next we encode the coefficients. There is no reason to encrypt these since they
	# are not private data.
	encoded_coefficients = []
	encoded_coefficients_string = "Encoding plaintext coefficients: "
	for i in range(10):
		encoded_coefficients.append(encoder.encode(coefficients[i]))
		encoded_coefficients_string += (str)(coefficients[i])[:6]
		if i < 9: encoded_coefficients_string += ", "
	print(encoded_coefficients_string)

	# We also need to encode 0.1. Multiplication by this plaintext will have the
	# effect of dividing by 10. Note that in SEAL it is impossible to divide
	# ciphertext by another ciphertext, but in this way division by a plaintext is
	# possible.
	div_by_ten = encoder.encode(0.1)

	# Now compute each multiplication.

	print("Computing products: ")
	for i in range(10):
		# Note how we use plain multiplication instead of usual multiplication. The
		# result overwrites the first argument in the function call.
		evaluator.multiply_plain(encrypted_rationals[i], encoded_coefficients[i])
	print("Done")

	# To obtain the linear sum we need to still compute the sum of the ciphertexts
	# in encrypted_rationals. There is an easy way to add together a vector of
	# Ciphertexts.
	encrypted_result = Ciphertext()
	print("Adding up all 10 ciphertexts: ")
	evaluator.add_many(encrypted_rationals, encrypted_result)
	print("Done")

	# Perform division by 10 by plain multiplication with div_by_ten.
	print("Dividing by 10: ")
	evaluator.multiply_plain(encrypted_result, div_by_ten)
	print("Done")

	# How much noise budget do we have left?
	print("Noise budget in result: " + (str)(decryptor.invariant_noise_budget(encrypted_result)) + " bits")

	# Decrypt, decode, and print result.
	plain_result = Plaintext()
	print("Decrypting result: ")
	decryptor.decrypt(encrypted_result, plain_result)
	print("Done")
	result = encoder.decode(plain_result)
	print("Weighted average: " + (str)(result)[:8])


def example_batching():
	print_example_banner("Example: Batching with PolyCRTBuilder");

	# In this fundamental example we discuss and demonstrate a powerful technique
	# called `batching'. If N denotes the degree of the polynomial modulus, and T
	# the plaintext modulus, then batching is automatically enabled in SEAL if
	# T is a prime and congruent to 1 modulo 2*N. In batching the plaintexts are
	# viewed as matrices of size 2-by-(N/2) with each element an integer modulo T.
	# Homomorphic operations act element-wise between encrypted matrices, allowing
	# the user to obtain speeds-ups of several orders of magnitude in naively
	# vectorizable computations. We demonstrate two more homomorphic operations
	# which act on encrypted matrices by rotating the rows cyclically, or rotate
	# the columns (i.e. swap the rows). These operations require the construction
	# of so-called `Galois keys', which are very similar to evaluation keys.
	parms = EncryptionParameters()

	parms.set_poly_modulus("1x^4096 + 1")
	parms.set_coeff_modulus(seal.coeff_modulus_128(4096))

	# Note that 40961 is a prime number and 2*4096 divides 40960.
	parms.set_plain_modulus(40961)

	context = SEALContext(parms)
	print_parameters(context)

	# We can see that batching is indeed enabled by looking at the encryption
	# parameter qualifiers created by SEALContext.
	qualifiers = context.qualifiers()
	#print("Batching enable: " + boolalpha + qualifiers.enable_batching)
	#print("Batching enable: " + qualifiers.enable_batching)

	keygen = KeyGenerator(context)
	public_key = keygen.public_key()
	secret_key = keygen.secret_key()

	# We need to create Galois keys for performing matrix row and column rotations.
	# Like evaluation keys, the behavior of Galois keys depends on a decomposition
	# bit count. The noise budget consumption behavior of matrix row and column
	# rotations is exactly like that of relinearization. Thus, we refer the reader
	# to example_basics_ii() for more details.

	# Here we use a moderate size decomposition bit count.
	gal_keys = GaloisKeys()
	keygen.generate_galois_keys(30, gal_keys)

	# Since we are going to do some multiplications we will also relinearize.
	ev_keys = EvaluationKeys()
	keygen.generate_evaluation_keys(30, ev_keys)

	# We also set up an Encryptor, Evaluator, and Decryptor here.
	encryptor = Encryptor(context, public_key)
	evaluator = Evaluator(context)
	decryptor = Decryptor(context, secret_key)

	# Batching is done through an instance of the PolyCRTBuilder class so need
	# to start by constructing one.
	crtbuilder = PolyCRTBuilder(context)

	# The total number of batching `slots' is degree(poly_modulus). The matrices
	# we encrypt are of size 2-by-(slot_count / 2).
	slot_count = (int)(crtbuilder.slot_count())
	row_size = (int)(slot_count / 2)
	print("Plaintext matrix row size: " + (str)(row_size))

	# Printing the matrix is a bit of a pain.
	def print_matrix(matrix):
		print("")

		# We're not going to print every column of the matrix (there are 2048). Instead
		# print this many slots from beginning and end of the matrix.
		print_size = 5

		current_line = "    ["
		for i in range(print_size):
			current_line += ((str)(matrix[i]) + ", ")
		current_line += ("..., ")
		for i in range(row_size - print_size, row_size):
			current_line += ((str)(matrix[i]))
			if i != row_size-1: current_line += ", "
			else: current_line += "]"
		print(current_line)

		current_line = "    ["
		for i in range(row_size, row_size + print_size):
			current_line += ((str)(matrix[i]) + ", ")
		current_line += ("..., ")
		for i in range(2*row_size - print_size, 2*row_size):
			current_line += ((str)(matrix[i]))
			if i != 2*row_size-1: current_line += ", "
			else: current_line += "]"
		print(current_line)
		print("")

	# The matrix plaintext is simply given to PolyCRTBuilder as a flattened vector
	# of numbers of size slot_count. The first row_size numbers form the first row,
	# and the rest form the second row. Here we create the following matrix:

	#     [ 0,  1,  2,  3,  0,  0, ...,  0 ]
	#     [ 4,  5,  6,  7,  0,  0, ...,  0 ]
	pod_matrix = [0]*slot_count
	pod_matrix[0] = 0
	pod_matrix[1] = 1
	pod_matrix[2] = 2
	pod_matrix[3] = 3
	pod_matrix[row_size] = 4
	pod_matrix[row_size + 1] = 5
	pod_matrix[row_size + 2] = 6
	pod_matrix[row_size + 3] = 7

	print("Input plaintext matrix:")
	print_matrix(pod_matrix)

	# First we use PolyCRTBuilder to compose the matrix into a plaintext.
	plain_matrix = Plaintext()
	crtbuilder.compose(pod_matrix, plain_matrix)

	# Next we encrypt the plaintext as usual.
	encrypted_matrix = Ciphertext()
	print("Encrypting: ")
	encryptor.encrypt(plain_matrix, encrypted_matrix)
	print("Done")
	print("Noise budget in fresh encryption: " +
	      (str)(decryptor.invariant_noise_budget(encrypted_matrix)) + " bits")

	# Operating on the ciphertext results in homomorphic operations being performed
	# simultaneously in all 4096 slots (matrix elements). To illustrate this, we
	# form another plaintext matrix

	#     [ 1,  2,  1,  2,  1,  2, ..., 2 ]
	#     [ 1,  2,  1,  2,  1,  2, ..., 2 ]

	# and compose it into a plaintext.
	pod_matrix2 = []
	for i in range(slot_count): pod_matrix2.append((i % 2) + 1)
	plain_matrix2 = Plaintext()
	crtbuilder.compose(pod_matrix2, plain_matrix2)
	print("Second input plaintext matrix:")
	print_matrix(pod_matrix2)

	# We now add the second (plaintext) matrix to the encrypted one using another
	# new operation -- plain addition -- and square the sum.
	print("Adding and squaring: ")
	evaluator.add_plain(encrypted_matrix, plain_matrix2)
	evaluator.square(encrypted_matrix)
	evaluator.relinearize(encrypted_matrix, ev_keys)
	print("Done")

	# How much noise budget do we have left?
	print("Noise budget in result: " + (str)(decryptor.invariant_noise_budget(encrypted_matrix)) + " bits")

	# We decrypt and decompose the plaintext to recover the result as a matrix.
	plain_result = Plaintext()
	print("Decrypting result: ")
	decryptor.decrypt(encrypted_matrix, plain_result)
	print("Done")

	crtbuilder.decompose(plain_result)
	pod_result = [plain_result.coeff_at(i) for i in range(plain_result.coeff_count())]

	print("Result plaintext matrix:")
	print_matrix(pod_result)

	# Note how the operation was performed in one go for each of the elements of the
	# matrix. It is possible to achieve incredible performance improvements by using
	# this method when the computation is easily vectorizable.

	# All of our discussion so far could have applied just as well for a simple vector
	# data type (not matrix). Now we show how the matrix view of the plaintext can be
	# used for more functionality. Namely, it is possible to rotate the matrix rows
	# cyclically, and same for the columns (i.e. swap the two rows). For this we need
	# the Galois keys that we generated earlier.

	# We return to the original matrix that we started with.
	encryptor.encrypt(plain_matrix, encrypted_matrix)
	print("Unrotated matrix: ")
	print_matrix(pod_matrix)
	print("Noise budget in fresh encryption: " +
	      (str)(decryptor.invariant_noise_budget(encrypted_matrix)) + " bits")

	# Now rotate the rows to the left 3 steps, decrypt, decompose, and print.
	evaluator.rotate_rows(encrypted_matrix, 3, gal_keys)
	print("Rotated rows 3 steps left: ")
	decryptor.decrypt(encrypted_matrix, plain_result)
	crtbuilder.decompose(plain_result)
	pod_result = [plain_result.coeff_at(i) for i in range(plain_result.coeff_count())]
	print_matrix(pod_result)
	print("Noise budget after rotation" +
	      (str)(decryptor.invariant_noise_budget(encrypted_matrix)) + " bits")

	# Rotate columns (swap rows), decrypt, decompose, and print.
	evaluator.rotate_columns(encrypted_matrix, gal_keys)
	print("Rotated columns: ")
	decryptor.decrypt(encrypted_matrix, plain_result)
	crtbuilder.decompose(plain_result)
	pod_result = [plain_result.coeff_at(i) for i in range(plain_result.coeff_count())]
	print_matrix(pod_result)
	print("Noise budget after rotation: " +
	      (str)(decryptor.invariant_noise_budget(encrypted_matrix)) + " bits")

	# Rotate rows to the right 4 steps, decrypt, decompose, and print.
	evaluator.rotate_rows(encrypted_matrix, -4, gal_keys)
	print("Rotated rows 4 steps right: ")
	decryptor.decrypt(encrypted_matrix, plain_result)
	crtbuilder.decompose(plain_result)
	pod_result = [plain_result.coeff_at(i) for i in range(plain_result.coeff_count())]
	print_matrix(pod_result)
	print("Noise budget after rotation: " +
	      (str)(decryptor.invariant_noise_budget(encrypted_matrix)) + " bits")

	# The output is as expected. Note how the noise budget gets a big hit in the
	# first rotation, but remains almost unchanged in the next rotations. This is
	# again the same phenomenon that occurs with relinearization, where the noise
	# budget is consumed down to some bound determined by the decomposition bit count
	# and the encryption parameters. For example, after some multiplications have
	# been performed, rotations might practically for free (noise budget-wise), but
	# might be relatively expensive when the noise budget is nearly full, unless
	# a small decomposition bit count is used, which again is computationally costly.


def example_parameter_selection():
	print_example_banner("Example: Automatic Parameter Selection")

	# SEAL contains an automatic parameter selection tool that can help the user
	# select optimal parameters that support a particular computation. In this
	# example we show how the tool can be used to find parameters for evaluating
	# the degree 3 polynomial 42x^3-27x+1 on an encrypted input encoded with the
	# IntegerEncoder. For this to be possible, we need to know an upper bound on
	# the size of the input, and in this example assume that x is an integer with
	# base-3 representation of length at most 10.
	print("Finding optimized parameters for computing 42x^3-27x+1: ")

	# The set of tools in the parameter selector are ChooserPoly, ChooserEvaluator,
	# ChooserEncoder, ChooserEncryptor, and ChooserDecryptor. Of these the most
	# important ones are ChooserPoly, which is an object representing the input
	# data both in plaintext and encrypted form, and ChooserEvaluator, which
	# simulates plaintext coefficient growth and noise budget consumption in the
	# computations. Here we use also the ChooserEncoder to conveniently obtain
	# ChooserPoly objects modeling the plaintext coefficients 42, -27, and 1.

	# Note that we are using the IntegerEncoder with base 3.
	chooser_encoder = ChooserEncoder(3)
	chooser_evaluator = ChooserEvaluator()

	# First we create a ChooserPoly representing the input data. You can think of
	# this modeling a freshly encrypted ciphertext of a plaintext polynomial of
	# length at most 10 coefficients, where the coefficients have absolute value
	# at most 1 (as is the case when using IntegerEncoder with base 3).
	c_input = ChooserPoly(10, 1)

	# Normally Evaluator::exponentiate takes the evaluation keys as argument. Since
	# no keys exist here, we simply pass the desired decomposition bit count (15)
	# to the ChooserEvaluator::exponentiate function.

	# Here we compute the first term.
	c_cubed_input = chooser_evaluator.exponentiate(c_input, 3, 15)
	c_term1 = chooser_evaluator.multiply_plain(c_cubed_input, chooser_encoder.encode(42))

	# Then compute the second term.
	c_term2 = chooser_evaluator.multiply_plain(c_input, chooser_encoder.encode(27))

	# Subtract the first two terms.
	c_sum12 = chooser_evaluator.sub(c_term1, c_term2)

	# Finally add a plaintext constant 1.
	c_result = chooser_evaluator.add_plain(c_sum12, chooser_encoder.encode(1))

	# The optimal parameters are now computed using the select_parameters
	# function in ChooserEvaluator. It is possible to give this function the
	# results of several distinct computations (as ChooserPoly objects), all
	# of which are supposed to be possible to perform with the resulting set
	# of parameters. However, here we have only one input ChooserPoly.
	optimal_parms = EncryptionParameters()
	chooser_evaluator.select_parameters([c_result], 0, optimal_parms)
	print("Done")

	# Create an SEALContext object for the returned parameters
	optimal_context = SEALContext(optimal_parms)
	print_parameters(optimal_context)

	# Do the parameters actually make any sense? We can try to perform the
	# homomorphic computation using the given parameters and see what happens.
	keygen = KeyGenerator(optimal_context)
	public_key = keygen.public_key()
	secret_key = keygen.secret_key()
	ev_keys = EvaluationKeys()
	keygen.generate_evaluation_keys(15, ev_keys)

	encryptor = Encryptor(optimal_context, public_key)
	evaluator = Evaluator(optimal_context)
	decryptor = Decryptor(optimal_context, secret_key)
	encoder = IntegerEncoder(optimal_context.plain_modulus(), 3)

	# Now perform the computations on some real data.
	input_value = 12345
	plain_input = encoder.encode(input_value)
	print("Encoded " + (str)(input_value) + " as polynomial " + plain_input.to_string())

	input_ciphertext = Ciphertext()
	print("Encrypting: ")
	encryptor.encrypt(plain_input, input_ciphertext)
	print("Done")

	print("Computing 42x^3-27x+1 on encrypted x=12345: ")
	deg3_term = Ciphertext()
	evaluator.exponentiate(input_ciphertext, 3, ev_keys, deg3_term)
	evaluator.multiply_plain(deg3_term, encoder.encode(42))
	deg1_term = Ciphertext()
	evaluator.multiply_plain(input_ciphertext, encoder.encode(27), deg1_term)
	evaluator.sub(deg3_term, deg1_term)
	evaluator.add_plain(deg3_term, encoder.encode(1))
	print("Done")

	# Now deg3_term holds the result. We decrypt, decode, and print the result.
	plain_result = Plaintext()
	print("Decrypting: ")
	decryptor.decrypt(deg3_term, plain_result)
	print("Done")
	print("Polynomial 42x^3-27x+1 evaluated at x=12345: " + (str)(encoder.decode_int64(plain_result)))

	# We should have a reasonable amount of noise room left if the parameter
	# selection was done properly. The user can experiment for instance by
	# changing the decomposition bit count, and observing how it affects the
	# result. Typically the budget should never be even close to 0. Instead,
	# SEAL uses heuristic upper bound estimates on the noise budget consumption,
	# which ensures that the computation will succeed with very high probability
	# with the selected parameters.
	print("Noise budget in result: " + (str)(decryptor.invariant_noise_budget(deg3_term)) + " bits")


def example_performance_st():
	print_example_banner("Example: Performance Test (Single Thread)");

	# In this thread we perform a timing of basic operations in single-threaded
	# execution. We use the following lambda function to run the timing example.
	def performance_test_st(context):
		print_parameters(context)
		poly_modulus = context.poly_modulus()
		coeff_modulus = context.total_coeff_modulus()
		plain_modulus = context.plain_modulus()

		# Set up keys. For both relinearization and rotations we use a large
		# decomposition bit count for best possible computational performance.
		dbc = seal.dbc_max()
		print("Generating secret/public keys: ")
		keygen = KeyGenerator(context)
		print("Done")

		secret_key = keygen.secret_key()
		public_key = keygen.public_key()

		# Generate evaluation keys.
		ev_keys = EvaluationKeys()
		print("Generating evaluation keys (dbc = " + (str)(dbc) + "): ")
		time_start = time.time()
		keygen.generate_evaluation_keys(dbc, ev_keys)
		time_end = time.time()
		time_diff = time_end - time_start
		print("Done [" + (str)(1000000*time_diff) + " microseconds]")

		# Generate Galois keys. In larger examples the Galois keys can use
		# a significant amount of memory, which can become a problem in constrained
		# systems. The user should try enabling some of the larger runs of
		# the test (see below) and to observe their effect on the memory pool
		# allocation size. The key generation can also take a significant
		# amount of time, as can be observed from the print-out.
		gal_keys = GaloisKeys()
		"""if not context.qualifiers().enable_batching:
			print("Given encryption parameters do not support batching.")
			return"""
		print("Generating Galois keys (dbc = " + (str)(dbc) + "): ")
		time_start = time.time()
		keygen.generate_galois_keys(dbc, gal_keys)
		time_end = time.time()
		time_diff = time_end - time_start
		print("Done [" + (str)(1000000*time_diff) + " microseconds]")

		encryptor = Encryptor(context, public_key)
		decryptor = Decryptor(context, secret_key)
		evaluator = Evaluator(context)
		crtbuilder = PolyCRTBuilder(context)
		encoder = IntegerEncoder(plain_modulus)

		# These will hold the total times used by each operation.
		time_batch_sum = 0
		time_unbatch_sum = 0
		time_encrypt_sum = 0
		time_decrypt_sum = 0
		time_add_sum = 0
		time_multiply_sum = 0
		time_multiply_plain_sum = 0
		time_square_sum = 0
		time_relinearize_sum = 0
		time_rotate_rows_one_step_sum = 0
		time_rotate_rows_random_sum = 0
		time_rotate_columns_sum = 0

		# How many times to run the test?
		count = 10

		# Populate a vector of values to batch.
		pod_vector = []
		for i in range(crtbuilder.slot_count()):
			pod_vector.append(random.randint(0, 4294967295) % plain_modulus.value())

		print("Running tests ")
		for i in range(count):
			# [Batching]
			# There is nothing unusual here. We batch our random plaintext matrix into
			# the polynomial. The user can try changing the decomposition bit count to
			# something smaller to see the effect. Note that the plaintext we use is of
			# the correct size, so no unnecessary reallocations are needed.
			plain = Plaintext(context.parms().poly_modulus().coeff_count(), 0)
			time_start = time.time()
			crtbuilder.compose(pod_vector, plain)
			time_end = time.time()
			time_batch_sum += (time_end - time_start)

			# [Unbatching]
			# We unbatch what we just batched.
			pod_vector2 = []
			time_start = time.time()
			decomposed_plain = plain
			crtbuilder.decompose(decomposed_plain)
			pod_vector2 = [decomposed_plain.coeff_at(i) for i in range(decomposed_plain.coeff_count())]
			time_end = time.time()
			time_unbatch_sum += (time_end - time_start)
			if pod_vector2 != pod_vector:
				print("Batch/unbatch failed. Something is wrong.")
				return

			# [Encryption]
			# We make sure our ciphertext is already allocated and large enough to
			# hold the encryption with these encryption parameters. We encrypt our
			# random batched matrix here.
			encrypted = Ciphertext(context.parms())
			time_start = time.time()
			encryptor.encrypt(plain, encrypted)
			time_end = time.time()
			time_encrypt_sum += (time_end - time_start)

			# [Decryption]
			# We decrypt what we just encrypted.
			plain2 = Plaintext(context.parms().poly_modulus().coeff_count(), 0)
			time_start = time.time()
			decryptor.decrypt(encrypted, plain2)
			time_end = time.time()
			time_decrypt_sum += (time_end - time_start)
			if plain2.to_string() != plain.to_string():
				print((str)(plain2.coeff_count()) + " " + (str)(plain2.significant_coeff_count()))
				print("Encrypt/decrypt failed. Something is wrong.")
				return

			# [Add]
			# We create two ciphertexts that are both of size 2, and perform a few
			# additions with them.
			encrypted1 = Ciphertext(context.parms())
			encryptor.encrypt(encoder.encode(i), encrypted1)
			encrypted2 = Ciphertext(context.parms())
			encryptor.encrypt(encoder.encode(i + 1), encrypted2)
			time_start = time.time()
			evaluator.add(encrypted1, encrypted1)
			evaluator.add(encrypted2, encrypted2)
			evaluator.add(encrypted1, encrypted2)
			time_end = time.time()
			time_add_sum += (time_end - time_start) / 3.0

			# [Multiply]
			# We multiply two ciphertexts of size 2. Since the size of the result
			# will be 3, and will overwrite the first argument, we reserve first
			# enough memory to avoid reallocating during multiplication.
			encrypted1.reserve(3)
			time_start = time.time()
			evaluator.multiply(encrypted1, encrypted2)
			time_end = time.time()
			time_multiply_sum += (time_end - time_start)

			# [Multiply Plain]
			# We multiply a ciphertext of size 2 with a random plaintext. Recall
			# that plain multiplication does not change the size of the ciphertext,
			# so we use encrypted2 here, which still has size 2.
			time_start = time.time()
			evaluator.multiply_plain(encrypted2, plain)
			time_end = time.time()
			time_multiply_plain_sum += (time_end - time_start)

			# [Square]
			# We continue to use the size 2 ciphertext encrypted2. Now we square it
			# which is a faster special case of homomorphic multiplication.
			time_start = time.time()
			evaluator.square(encrypted2)
			time_end = time.time()
			time_square_sum += (time_end - time_start)

			# [Relinearize]
			# Time to get back to encrypted1, which at this point is still of size 3.
			# We now relinearize it back to size 2. Since the allocation is currently
			# big enough to contain a ciphertext of size 3, no reallocation is needed
			# in the process.
			time_start = time.time()
			evaluator.relinearize(encrypted1, ev_keys)
			time_end = time.time()
			time_relinearize_sum += (time_end - time_start)

			# [Rotate Rows One Step]
			# We rotate matrix rows by one step left and measure the time.
			time_start = time.time()
			evaluator.rotate_rows(encrypted, 1, gal_keys)
			evaluator.rotate_rows(encrypted, -1, gal_keys)
			time_end = time.time()
			time_rotate_rows_one_step_sum += (time_end - time_start) / 2

			# [Rotate Rows Random]
			# We rotate matrix rows by a random number of steps. This is a bit more
			# expensive than rotating by just one step.
			row_size = crtbuilder.slot_count() / 2
			random_rotation = (int)((int)(random.random()*(row_size+1)) % row_size)
			time_start = time.time()
			evaluator.rotate_rows(encrypted, random_rotation, gal_keys)
			time_end = time.time()
			time_rotate_rows_random_sum += (time_end - time_start)

			# [Rotate Columns]
			# Nothing surprising here.
			time_start = time.time()
			evaluator.rotate_columns(encrypted, gal_keys)
			time_end = time.time()
			time_rotate_columns_sum += (time_end - time_start)

			# Print a dot to indicate progress.
			print(".")

		print(" Done")

		avg_batch = (time_batch_sum / count)*1000000
		avg_unbatch = (time_unbatch_sum / count)*1000000
		avg_encrypt = (time_encrypt_sum / count)*1000000
		avg_decrypt = (time_decrypt_sum / count)*1000000
		avg_add = (time_add_sum / count)*1000000
		avg_multiply = (time_multiply_sum / count)*1000000
		avg_multiply_plain = (time_multiply_plain_sum / count)*1000000
		avg_square = (time_square_sum / count)*1000000
		avg_relinearize = (time_relinearize_sum / count)*1000000
		avg_rotate_rows_one_step = (time_rotate_rows_one_step_sum / count)*1000000
		avg_rotate_rows_random = (time_rotate_rows_random_sum / count)*1000000
		avg_rotate_columns = (time_rotate_columns_sum / count)*1000000

		print("Average batch: " + (str)(avg_batch) + " microseconds")
		print("Average unbatch: " + (str)(avg_unbatch) + " microseconds")
		print("Average encrypt: " + (str)(avg_encrypt) + " microseconds")
		print("Average decrypt: " + (str)(avg_decrypt) + " microseconds")
		print("Average add: " + (str)(avg_add) + " microseconds")
		print("Average multiply: " + (str)(avg_multiply) + " microseconds")
		print("Average multiply plain: " + (str)(avg_multiply_plain) + " microseconds")
		print("Average square: " + (str)(avg_square) + " microseconds")
		print("Average relinearize: " + (str)(avg_relinearize) + " microseconds")
		print("Average rotate rows one step: " + (str)(avg_rotate_rows_one_step) + " microseconds")
		print("Average rotate rows random: " + (str)(avg_rotate_rows_random) + " microseconds")
		print("Average rotate columns: " + (str)(avg_rotate_columns) + " microseconds")

	parms = EncryptionParameters()
	parms.set_poly_modulus("1x^4096 + 1")
	parms.set_coeff_modulus(seal.coeff_modulus_128(4096))
	parms.set_plain_modulus(786433)
	context = SEALContext(parms)
	performance_test_st(context)

	print("")
	parms.set_poly_modulus("1x^8192 + 1")
	parms.set_coeff_modulus(seal.coeff_modulus_128(8192))
	parms.set_plain_modulus(786433)
	context = SEALContext(parms)
	performance_test_st(context)

	# Comment out the following to run the bigger examples.
	"""print("")
	parms.set_poly_modulus("1x^16384 + 1")
	parms.set_coeff_modulus(seal.coeff_modulus_128(16384))
	parms.set_plain_modulus(786433)
	context = SEALContext(parms)
	performance_test_st(context)

	print("")
	parms.set_poly_modulus("1x^32768 + 1")
	parms.set_coeff_modulus(seal.coeff_modulus_128(32768))
	parms.set_plain_modulus(786433)
	context = SEALContext(parms)
	performance_test_st(context)"""


def example_performance_mt(th_count):
	print_example_banner("Example: Performance Test (" + (str)(th_count) + " Threads)")

	#In this example we show how to efficiently run SEAL in a multi-threaded
	# application.

	# SEAL does not use multi-threading inside its functions, but most of the
	# tools such as Encryptor, Decryptor, PolyCRTBuilder, and Evaluator are by
	# default thread-safe. However, by default these classes perform a large
	# number of allocations from a thread-safe memory pool, which can get slow
	# when several threads are used. Instead, here we show how the user can create
	# local memory pools using the MemoryPoolHandle class, which can be either
	# thread-safe (slower) or thread-unsafe (faster). For example, here we use
	# the MemoryPoolHandle class to essentially get thread-local memory pools.

	# First we set up shared instances of EncryptionParameters, SEALContext,
	# KeyGenerator, keys, Encryptor, Decryptor, Evaluator, PolyCRTBuilder.
	# After these classes are constructed, they are thread-safe to use.
	parms = EncryptionParameters()
	parms.set_poly_modulus("1x^8192 + 1")
	parms.set_coeff_modulus(seal.coeff_modulus_128(8192))
	parms.set_plain_modulus(786433)

	context = SEALContext(parms)
	print_parameters(context)

	poly_modulus = context.poly_modulus()
	coeff_modulus = context.total_coeff_modulus()
	plain_modulus = context.plain_modulus()

	dbc = seal.dbc_max()
	print("Generating secret/public keys: ")
	keygen = KeyGenerator(context)
	print("Done")

	secret_key = keygen.secret_key()
	public_key = keygen.public_key()

	ev_keys = EvaluationKeys()
	print("Generating evaluation keys (dbc = " + (str)(dbc) + "): ")
	time_start = time.time()
	keygen.generate_evaluation_keys(dbc, ev_keys)
	time_end = time.time()
	time_diff = (time_end - time_start)*1000000
	print("Done [" + (str)(time_diff) + " microseconds]")

	gal_keys = GaloisKeys()
	"""if not context.qualifiers().enable_batching:
		print("Given encryption parameters do not support batching.")
		return"""
	print("Generating Galois keys (dbc = " + (str)(dbc) + "): ")
	time_start = time.time()
	keygen.generate_galois_keys(dbc, gal_keys)
	time_end = time.time()
	time_diff = (time_end - time_start)*1000000
	print("Done [" + (str)(time_diff) + " microseconds]")

	encryptor = Encryptor(context, public_key)
	decryptor = Decryptor(context, secret_key)
	evaluator = Evaluator(context)
	crtbuilder = PolyCRTBuilder(context)
	encoder = IntegerEncoder(plain_modulus)

	# We need a lambda function similar to what we had in the single-threaded
	# performance example. In this case the functions is slightly different,
	# since we share the same SEALContext, other tool classes, and keys among
	# all threads (captured by reference below). We also take a MemoryPoolHandle
	# as an argument; the memory pool managed by this MemoryPoolHandle will be
	# used for all dynamic allocations in the homomorphic computations.
	def performance_test_mt(th_index, lock, pool):
		# Print the thread index and memory pool address. The idea is that for
		# each thread we pass a MemoryPoolHandle pointing to a new memory pool.
		# The given MemoryPoolHandle is then used for all allocations inside this
		# function, and all functions it calls, e.g. plaintext and ciphertext
		# allocations, and allocations that occur during homomorphic operations.
		# This prevents costly concurrent allocations from becoming a bottleneck.
		lock.acquire()
		print("")
		print("Thread index: " + (str)(th_index))
		#cout << "Memory pool address: " << hex
		#    << "0x" << &pool.operator seal::util::MemoryPool &() << dec << endl;
		print("Starting tests ... ")
		lock.release()

		poly_modulus = context.poly_modulus()
		coeff_modulus = context.total_coeff_modulus()
		plain_modulus = context.plain_modulus()

		# These will hold the total times used by each operation.
		time_batch_sum = 0
		time_unbatch_sum = 0
		time_encrypt_sum = 0
		time_decrypt_sum = 0
		time_add_sum = 0
		time_multiply_sum = 0
		time_multiply_plain_sum = 0
		time_square_sum = 0
		time_relinearize_sum = 0
		time_rotate_rows_one_step_sum = 0
		time_rotate_rows_random_sum = 0
		time_rotate_columns_sum = 0

		# How many times to run the test?
		count = 10

		# Populate a vector of values to batch if enable_batching == true.
		pod_vector = []
		for i in range(crtbuilder.slot_count()):
			pod_vector.append(random.randint(0, 4294967295) % plain_modulus.value())

		for i in range(count):
			# [Batching]
			# Note that we pass the MemoryPoolHandle as an argment to the constructor
			# of the plaintext. This was the plaintext memory is allocated from the
			# thread-local memory pool, and costly concurrent allocations from the
			# same memory pool can be avoided.
			plain = Plaintext(context.parms().poly_modulus().coeff_count(), 0, pool)
			time_start = time.time()
			crtbuilder.compose(pod_vector, plain)
			time_end = time.time()
			time_batch_sum += (time_end - time_start)*1000000

			# [Unbatching]
			# The decompose-operation needs to perform a single allocation from
			# a memory pool. Note how we pass our MemoryPoolHandle to it as an
			# argument, suggesting it to use the given pool for the allocation.
			# Again, we avoid having several threads allocating from the same
			# memory pool concurrently.
			time_start = time.time()
			crtbuilder.decompose(plain, pool)
			pod_vector2 = [plain.coeff_at(i) for i in range(plain.coeff_count())]
			time_end = time.time()
			time_unbatch_sum += (time_end - time_start)*1000000
			if pod_vector2 != pod_vector:
				print("Batch/unbatch failed. Something is wrong.")
				return

			# [Encryption]
			# We allocate the result ciphertext from the local memory pool. Here
			#encryption also takes the MemoryPoolHandle as an argument.
			encrypted = Ciphertext(context.parms(), pool)
			time_start = time.time()
			encryptor.encrypt(plain, encrypted, pool)
			time_end = time.time()
			time_encrypt_sum += (time_end - time_start)*1000000

			# [Decryption]
			plain2 = Plaintext(context.parms().poly_modulus().coeff_count(), 0, pool)
			time_start = time.time()
			decryptor.decrypt(encrypted, plain2, pool)
			time_end = time.time()
			time_decrypt_sum += (time_end - time_start)*1000000
			if plain2.to_string() != plain.to_string():
				print((str)(plain2.coeff_count()) + " " + (str)(plain2.significant_coeff_count()))
				print((str)(plain.coeff_count()) + " " + (str)(plain.significant_coeff_count()))
				print("Encrypt/decrypt failed. Something is wrong.")
				return

			# [Add]
			# Note how both ciphertexts are allocated from the local memory pool, and
			# how the local memory pool is also used for encryption. Homomorphic
			# addition on the other hand does not need to make any dynamic allocations.
			encrypted1 = Ciphertext(context.parms(), pool)
			encryptor.encrypt(encoder.encode(i), encrypted1, pool)
			encrypted2 = Ciphertext(context.parms(), pool)
			encryptor.encrypt(encoder.encode(i + 1), encrypted2, pool)
			time_start = time.time()
			evaluator.add(encrypted1, encrypted1)
			evaluator.add(encrypted2, encrypted2)
			evaluator.add(encrypted1, encrypted2)
			time_end = time.time()
			time_add_sum += ((time_end - time_start) / 3.0)*1000000

			# [Multiply]
			# Multiplication is a heavy-duty operation making several allocations
			# from the local memory pool.
			encrypted1.reserve(3)
			time_start = time.time()
			evaluator.multiply(encrypted1, encrypted2, pool)
			time_end = time.time()
			time_multiply_sum += (time_end - time_start)*1000000

			# [Multiply Plain]
			time_start = time.time()
			evaluator.multiply_plain(encrypted2, plain, pool)
			time_end = time.time()
			time_multiply_plain_sum += (time_end - time_start)*1000000

			# [Square]
			time_start = time.time()
			evaluator.square(encrypted2, pool)
			time_end = time.time()
			time_square_sum += (time_end - time_start)*1000000

			# [Relinearize]
			time_start = time.time()
			evaluator.relinearize(encrypted1, ev_keys, pool)
			time_end = time.time()
			time_relinearize_sum += (time_end - time_start)*1000000

			# [Rotate Rows One Step]
			time_start = time.time()
			evaluator.rotate_rows(encrypted, 1, gal_keys, pool)
			evaluator.rotate_rows(encrypted, -1, gal_keys, pool)
			time_end = time.time()
			time_rotate_rows_one_step_sum += ((time_end - time_start) / 2.0)*1000000

			# [Rotate Rows Random]
			row_size = crtbuilder.slot_count() / 2
			random_rotation = (int)((int)(random.random()*(row_size+1)) % row_size)
			time_start = time.time()
			evaluator.rotate_rows(encrypted, random_rotation, gal_keys, pool)
			time_end = time.time()
			time_rotate_rows_random_sum += (time_end - time_start)*1000000

			# [Rotate Columns]
			time_start = time.time()
			evaluator.rotate_columns(encrypted, gal_keys, pool)
			time_end = time.time()
			time_rotate_columns_sum += (time_end - time_start)*1000000

		avg_batch = time_batch_sum / count
		avg_unbatch = time_unbatch_sum / count
		avg_encrypt = time_encrypt_sum / count
		avg_decrypt = time_decrypt_sum / count
		avg_add = time_add_sum / count
		avg_multiply = time_multiply_sum / count
		avg_multiply_plain = time_multiply_plain_sum / count
		avg_square = time_square_sum / count
		avg_relinearize = time_relinearize_sum / count
		avg_rotate_rows_one_step = time_rotate_rows_one_step_sum / count
		avg_rotate_rows_random = time_rotate_rows_random_sum / count
		avg_rotate_columns = time_rotate_columns_sum / count

		lock.acquire()
		print("")
		print("Test finished for thread " + (str)(th_index))
		print("Average batch: " + (str)(avg_batch) + " microseconds")
		print("Average unbatch: " + (str)(avg_unbatch) + " microseconds")
		print("Average encrypt: " + (str)(avg_encrypt) + " microseconds")
		print("Average decrypt: " + (str)(avg_decrypt) + " microseconds")
		print("Average add: " + (str)(avg_add) + " microseconds")
		print("Average multiply: " + (str)(avg_multiply) + " microseconds")
		print("Average multiply plain: " + (str)(avg_multiply_plain) + " microseconds")
		print("Average square: " + (str)(avg_square) + " microseconds")
		print("Average relinearize: " + (str)(avg_relinearize) + " microseconds")
		print("Average rotate rows one step: " + (str)(avg_rotate_rows_one_step) + " microseconds")
		print("Average rotate rows random: " + (str)(avg_rotate_rows_random) + " microseconds")
		print("Average rotate columns: " + (str)(avg_rotate_columns) + " microseconds")
		lock.release()

	print_lock = threading.Lock()
	th_vector = []
	for i in range(th_count):
		# Each thread is created and given a MemoryPoolHandle pointing to a new
		# memory pool. Essentially, this results in thread-local memory pools
		# and resolves the contention that would result from several threads
		# allocating from e.g. the global memory pool. The bool argument given
		# to MemoryPoolHandle::New means that the created memory pool is
		# thread-unsafe, resulting in better performance. The user can change
		# the argument to "true" instead. However, in this small example the
		# difference in performance is non-existent.
		#new_pool = MemoryPoolHandle()
		#new_pool.
		th_vector.append(
			threading.Thread(target = performance_test_mt, args = (i + 1, print_lock, MemoryPoolHandle().New(False)))
		)

		# The global memory pool is thread-safe, and unless otherwise specified,
		# it is used for (nearly) all dynamic allocations. The user can comment
		# out the lines below to test the performance of the global memory pool
		# in this example. Again, the performance difference might only show up
		# when a large number of threads are used.
		"""th_vector.append(
			threading.Thread(target = performance_test_mt, args = (i + 1, print_lock, MemoryPoolHandle().acquire_global()))
		)"""


	# We are done here. Join the threads.
	for i in range(len(th_vector)):
		th_vector[i].start()

	for i in range(len(th_vector)):
		th_vector[i].join()



def save_example():

    print_example_banner("Example: Basics I");

   
    parms = EncryptionParameters()

    # We first set the polynomial modulus. This must be a power-of-2 cyclotomic
    # polynomial, i.e. a polynomial of the form "1x^(power-of-2) + 1". The polynomial
    # modulus should be thought of mainly affecting the security level of the scheme;
    # larger polynomial modulus makes the scheme more secure. At the same time, it
    # makes ciphertext sizes larger, and consequently all operations slower.
    # Recommended degrees for poly_modulus are 1024, 2048, 4096, 8192, 16384, 32768,
    # but it is also possible to go beyond this. Since we perform only a very small
    # computation in this example, it suffices to use a very small polynomial modulus
    parms.set_poly_modulus("1x^2048 + 1")

   
    parms.set_coeff_modulus(seal.coeff_modulus_128(2048))

    # The plaintext modulus can be any positive integer, even though here we take
    # it to be a power of two. In fact, in many cases one might instead want it to
    # be a prime number; we will see this in example_batching(). The plaintext
    # modulus determines the size of the plaintext data type, but it also affects
    # the noise budget in a freshly encrypted ciphertext, and the consumption of
    # the noise budget in homomorphic multiplication. Thus, it is essential to try
    # to keep the plaintext data type as small as possible for good performance.
    # The noise budget in a freshly encrypted ciphertext is

    #     ~ log2(coeff_modulus/plain_modulus) (bits)

    # and the noise budget consumption in a homomorphic multiplication is of the
    # form log2(plain_modulus) + (other terms).
    parms.set_plain_modulus(1 << 8)

    # Now that all parameters are set, we are ready to construct a SEALContext
    # object. This is a heavy class that checks the validity and properties of
    # the parameters we just set, and performs and stores several important
    # pre-computations.
    context = SEALContext(parms)

    # Print the parameters that we have chosen
    print_parameters(context);

 
    encoder = IntegerEncoder(context.plain_modulus())

    # We are now ready to generate the secret and public keys. For this purpose we need
    # an instance of the KeyGenerator class. Constructing a KeyGenerator automatically
    # generates the public and secret key, which can then be read to local variables.
    # To create a fresh pair of keys one can call KeyGenerator::generate() at any time.
    keygen = KeyGenerator(context)
    public_key = keygen.public_key()
    secret_key = keygen.secret_key()

    # To be able to encrypt, we need to construct an instance of Encryptor. Note that
    # the Encryptor only requires the public key.
    encryptor = Encryptor(context, public_key)

    # Computations on the ciphertexts are performed with the Evaluator class.
    evaluator = Evaluator(context)

    # We will of course want to decrypt our results to verify that everything worked,
    # so we need to also construct an instance of Decryptor. Note that the Decryptor
    # requires the secret key.
    decryptor = Decryptor(context, secret_key)

    # We start by encoding two integers as plaintext polynomials.
    value1 = 5;
    plain1 = encoder.encode(value1);
    print("Encoded " + (str)(value1) + " as polynomial " + plain1.to_string() + " (plain1)")

    value2 = -7;
    plain2 = encoder.encode(value2);
    print("Encoded " + (str)(value2) + " as polynomial " + plain2.to_string() + " (plain2)")

    # Encrypting the values is easy.
    encrypted1 = Ciphertext()
    encrypted2 = Ciphertext()
    print("Encrypting plain1: ")
    encryptor.encrypt(plain1, encrypted1)
    print("Done (encrypted1)")

    encrypted1.save()

    print("Encrypting plain2: ")
    encryptor.encrypt(plain2, encrypted2)
    print("Done (encrypted2)")

    # To illustrate the concept of noise budget, we print the budgets in the fresh
    # encryptions.
    print("Noise budget in encrypted1: " + (str)(decryptor.invariant_noise_budget(encrypted1)) + " bits")
    print("Noise budget in encrypted2: " + (str)(decryptor.invariant_noise_budget(encrypted2)) + " bits")

    # As a simple example, we compute (-encrypted1 + encrypted2) * encrypted2.

    # Negation is a unary operation.
    evaluator.negate(encrypted1)

    # Negation does not consume any noise budget.
    print("Noise budget in -encrypted1: " + (str)(decryptor.invariant_noise_budget(encrypted1)) + " bits")

    # Addition can be done in-place (overwriting the first argument with the result,
    # or alternatively a three-argument overload with a separate destination variable
    # can be used. The in-place variants are always more efficient. Here we overwrite
    # encrypted1 with the sum.
    evaluator.add(encrypted1, encrypted2)

    # It is instructive to think that addition sets the noise budget to the minimum
    # of the input noise budgets. In this case both inputs had roughly the same
    # budget going on, and the output (in encrypted1) has just slightly lower budget.
    # Depending on probabilistic effects, the noise growth consumption may or may
    # not be visible when measured in whole bits.
    print("Noise budget in -encrypted1 + encrypted2: " + (str)(decryptor.invariant_noise_budget(encrypted1)) + " bits")

    # Finally multiply with encrypted2. Again, we use the in-place version of the
    # function, overwriting encrypted1 with the product.
    evaluator.multiply(encrypted1, encrypted2)

    # Multiplication consumes a lot of noise budget. This is clearly seen in the
    # print-out. The user can change the plain_modulus to see its effect on the
    # rate of noise budget consumption.
    print("Noise budget in (-encrypted1 + encrypted2) * encrypted2: " + (str)(decryptor.invariant_noise_budget(encrypted1)) + " bits")

    # Now we decrypt and decode our result.
    plain_result = Plaintext()
    print("Decrypting result: ")
    decryptor.decrypt(encrypted1, plain_result)
    print("Done")

    # Print the result plaintext polynomial.
    print("Plaintext polynomial: " + plain_result.to_string())

    # Decode to obtain an integer result.
    print("Decoded integer: " + (str)(encoder.decode_int32(plain_result)))



def main():
	# Example: Basics I
	example_basics_i()
	# Example: Basics II
	example_basics_ii()
	# Example: How to pickle Ciphertexts
	example_pickle()
	# Example: Weighted Average
	example_weighted_average()
	# Example: Batching using CRT
	example_batching()
	# Example: Parameter Selection
	example_parameter_selection()
	# Example: Performance (Single Thread)
	example_performance_st()
	# Example: Performance (Multi Thread)
	th_count = ((int)(input('Thread count: ')))
	if th_count > 0: example_performance_mt(th_count)
	else: print('Invalid thread count.')
	# Wait for ENTER before closing screen.
	input('Press ENTER to exit')

def print_example_banner(title, ch='*', length=78):
	spaced_text = ' %s ' % title
	print(spaced_text.center(length, ch))

def print_parameters(context):
	print("/ Encryption parameters:")
	print("| poly_modulus: " + context.poly_modulus().to_string())

	# Print the size of the true (product) coefficient modulus
	print("| coeff_modulus_size: " + (str)(context.total_coeff_modulus().significant_bit_count()) + " bits")

	print("| plain_modulus: " + (str)(context.plain_modulus().value()))
	print("| noise_standard_deviation: " + (str)(context.noise_standard_deviation()))

if __name__ == '__main__':
	main()
