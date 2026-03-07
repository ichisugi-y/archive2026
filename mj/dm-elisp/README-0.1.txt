
dm-elisp: difference-based modules for emacs-lisp

					y-ichisugi@aist.go.jp
					2002-01-29

dm-elisp is an emacs-lisp with a very simple module mechanism,
that provides the following features:

	- No import/export declarations.
	- No name-collision due to the idea of fully-qualified-names.
	- Almost backward compatible with the traditional emacs-lisp. 
	- Supports information hiding.
	- Supports nested and/or overlapping name spaces 
		by means of multiple-inheritance of name spaces.

Motivation
----------
Difference-based module mechanism has been designed
 for object-oriented languages.
I have already implemented the module mechanism for Java. 
(See http://staff.aist.go.jp/y-ichisugi/mj/ )
However, the idea of this mechanism might be applicable to many other
programming languages.
In order to prove this, I implemented a prototyping module mechanism
for emacs-lisp.
Although the current implementation is not practical,
I think this module mechanism has a potential
to become the standard module mechanism of various languages,
including emacs-lisp.


The mechanism
-------------
Each file which contains programs written in dm-elisp is called "module". 
Modules are units of name space.
A module named "foo" is assumed to be written in the file "foo.el" .

The programmer of dm-elisp should use (dm-require "module-name")
instead of (require 'feature-name), if he want to use function or variable
names defined at the module.
You do not have to write (provide 'module-name) at the bottom of the file.

The required module is called super-module of the requiring module.
All names visible from the super-module are inherited to the sub-module.

Because a module can require more than one module,
the inheritance graph sometimes forms so called diamond inheritance,
as follows.

    A
  /   \
 B     C
  \   /
    D

If D refers to a name defined at A, there are no problem.
If B and C defines a name "foo" independently
and D refers to the name,
an ambiguous-reference error will be reported at the load-time of D.

When an ambiguous-reference error occurs,
the programmer can use FQN(fully-qualified-name) in order to avoid the error.
The FQN is denoted as "module-name:simple-name" .
For example, the name "foo" defined at the module B can
be referred to by "B:foo".

The dm-elisp programmer do not have to define names explicitly.
When the dm-require reads a new symbol which has not been interned 
in the current name space yet,
the symbol is regarded as a definition of the name.

When a name "foo" is defined at a module "M",
the name has a FQN, "M:foo", which can be referred to by
the all modules.

The end-user of the emacs should use FQNs instead of simple names.
For example, if the user want to invoke the function "foo"
define at the module M after loading the module,
he/she should type "M-x M:foo" instead of "M-x foo" .

The top-module is the super-module of all other modules.
All modules implicitly inherit the top-module.
It's name space is a snapshot of "obarray" when "(require 'dm-elisp)" is
invoked. 
Therefore, the name space of top-module contains
all names of pre-defined emacs-lisp functions and variables 
such as "defun", "cons" and "if".
These names have no FQN.

The function "intern" interns symbols into the name space of 
normal emacs-lisp.
You can use (dm-intern exp) or (dm-intern exp "module-name")
 instead of (intern exp), if necessary.


How to achieve information hiding
---------------------------------
If you want to separate public names and protected names,
you should separate a source file into three files.
The inheritance relation should be as follows:

     m1
     |
 m1-protected
     |
 m1-bottom


All public names should be defined at the module m1.
The module should contain only name definitions.
For example, if you want to make three names, foo, bar and baz
as public names, just write the following one s-expression
in the module in order to define these names:
	`(foo bar baz)

The implementation of the names should be defined at the module m1-protected.
The module should declare (dm-require "m1") at the top of the file.

The module m1-bottom is a module which only
 contains (dm-require "m1-protected") .

The end-users of emacs load this modules by (dm-require "m1-bottom"),
and use the public functions by typing as "M-x m1:foo" .

If you want define a module m2
which utilize the module m1 in the style of "block-box reuse",
you should make the inheritance relation as follows:

          m1
          |  \
m1-protected  \    m2
          |    \   |
   m1-bottom  m2-protected
            \      |
              m2-bottom

Because m2-protected does not inherit m1-protected,
the name space of m2-protected is not polluted by the protected names of m1.

The end-users of emacs load this modules by (dm-require "m2-bottom"),
and use the public functions by typing as "M-x m2:bar" .
Note that if m2-bottom does not inherit m1-bottom,
m1-protected and m1-bottom will not be loaded 
and the functions defined by m2 will not work.

If you dare to use the protected names of m1 when implementing m2,
that is "white-box reuse",
you may make the inheritance relation as follows:


          m1
          |  
m1-protected       m2
          |  \     |
   m1-bottom  m2-protected
            \      |
              m2-bottom


Nested and/or overlapping name spaces
-------------------------------------
Module inheritance represents nested name spaces.

  A
  |
  B

Because all names defined at the module A
are visible from the module B, 
the name-space of B can be considered as 
being inside of the name-space of A.

+---------------+
| A  +------+   |
|    | B    |   |
|    +------+   |
+---------------+

Multiple-inheritance of modules represents overlapping name spaces.
  
    A
  /   \
 B     C
  \   /
    D

Because the module D can refer all names defined at the B and C,
the module D can be considered as a inside of the overlapping name space
of B and C.

+-------------------+
| A +---------+     |
|   | B +-----+---+ |
|   |   | +-+ | C | |
|   |   | |D| |   | |
|   |   | +-+ |   | |
|   +---+-----+   | |
|       +---------+ |       
+-------------------+

Traditional module mechanism with import/export declarations
only provides a simple boundary which separates inside and outside of 
each module.

+--+  +--+  +--+  +--+  +--+
|m1|  |m2|  |m3|  |m4|  |m5|
+--+  +--+  +--+  +--+  +--+

In the case of difference-based module,
more general boundary can be expressed by means of 
the inheritance mechanism of name spaces.


How to run sample programs
--------------------------
;;;;Type as follows in the "*scratch*" buffer.

;;;--------------------------------------------------
;;; Load the ".elc" file of dm-elisp. Do NOT load the ".el" file.
(load-file "dm-elisp.elc") ; It takes some seconds...
t

;;;--------------------------------------------------
;;; Just a test program.
(dm-require "test3")
nil

;;; An example of white-box reuse and black-box reuse.
;;; Module "double-counter" uses "counter" in the style of white-box reuse.
;;; Module "test-counter" uses "double-counter" 
;;;in the style of black-box reuse.
;;; The inheritance graph is as follows:
;;;
;;;           counter
;;;              |   \
;;; counter-protected double-counter__________
;;;              |   \  |                     \
;;;    counter-bottom double-counter-protected \ test-counter
;;;                  \  |                       \  |
;;;                   double-counter-bottom___ test-counter-protected
;;;                                           \    |
;;;                                            test-counter-bottom
(dm-require "test-counter-bottom") ; It takes much more seconds...
nil
(test-counter:test-counter)

0

1

2

4
nil

;;;--------------------------------------------------
;;; An example of multiple-inheritance.
;;; Two functions foo1:foo and foo2:foo is defined independently.
;;; Modules call-foo1 and call-foo2 can access to these functions
;;;by simple-names because the reference is not ambiguous.
;;; The inheritance graph is as follows:
;;; 
;;;      foo1    foo2
;;;        |      |
;;; call-foo1    call-foo2
;;;          \  /
;;;       call-both-foo

(dm-require "call-both-foo")
nil
(call-both-foo:call-both-foo)

"Function foo defined at foo1.el is called."

"Function foo defined at foo2.el is called."
nil
(dm-intern "foo" 'foo1)
foo1:foo
(dm-intern "foo" 'foo2)
foo2:foo
(dm-intern "foo" 'call-foo1)
foo1:foo
(dm-intern "foo" 'call-foo2)
foo2:foo
(dm-intern "foo" 'call-both-foo); Ambiguous reference: (foo1:foo foo2:foo)


Implementation outline
----------------------

Modules are implemented as hashtables
whose keys are simple names and whose values are FQNs.
When dm-require is invoked, a new hashtable is created 
and it inherits all the names in the top-module.

During loading the required module,
 each s-expression read by emacs's "read" function
are translated before evaluation so that they contains appropriate FQNs.
For example, the following s-expression in a module m2

	(defun foo () (bar 'baz))

will be translated to the following before evaluation,

	(defun m2:foo () (m1:bar 'm2:baz))

if "bar" is a name defined at a super-module m1
and "foo" and "baz" are not defined at any super-modules.
The name "defun" is not changed because the name is
inherited from the top-module.

An ambiguous-reference error will be found and reported
during the name translation.

Just after the evaluation of all s-expressions in the required module,
all names in the required module (super-module)
are inherited to the requiring module (sub-module).


Limitations
-----------
Because current implementation is a prototyping system,
there are some limitations.

- Byte-code compilation is not supported.
- The variable load-path is not examined. Only the modules 
 under the current directory can be loaded.
- The dm-elisp program should not use "require", "load-file" or something 
 like that.
- Module inheritance is slow.
 (This may be improved by special treatment of top-module.)


Future work
-----------
- Better support for information hiding is necessary.
- This module mechanism should be implemented
by means of modification of source-code of emacs.
