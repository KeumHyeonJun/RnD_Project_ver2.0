??<
??
B
AssignVariableOp
resource
value"dtype"
dtypetype?
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
N
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype"
Truncatebool( 
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
p
GatherNd
params"Tparams
indices"Tindices
output"Tparams"
Tparamstype"
Tindicestype:
2	
?
HashTableV2
table_handle"
	containerstring "
shared_namestring "!
use_node_name_sharingbool( "
	key_dtypetype"
value_dtypetype?
.
Identity

input"T
output"T"	
Ttype
w
LookupTableFindV2
table_handle
keys"Tin
default_value"Tout
values"Tout"
Tintype"
Touttype?
b
LookupTableImportV2
table_handle
keys"Tin
values"Tout"
Tintype"
Touttype?
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(?

NoOp
U
NotEqual
x"T
y"T
z
"	
Ttype"$
incompatible_shape_errorbool(?
?
OneHot
indices"TI	
depth
on_value"T
	off_value"T
output"T"
axisint?????????"	
Ttype"
TItype0	:
2	
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype?
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
?
Select
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
0
Sigmoid
x"T
y"T"
Ttype:

2
?
SparseToDense
sparse_indices"Tindices
output_shape"Tindices
sparse_values"T
default_value"T

dense"T"
validate_indicesbool("	
Ttype"
Tindicestype:
2	
?
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ??
@
StaticRegexFullMatch	
input

output
"
patternstring
?
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
?
Sum

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?
E
Where

input"T	
index	"%
Ttype0
:
2	
"serve*2.7.02unknown8??8
?
sequential/dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*(
shared_namesequential/dense/kernel
?
+sequential/dense/kernel/Read/ReadVariableOpReadVariableOpsequential/dense/kernel* 
_output_shapes
:
??*
dtype0
?
sequential/dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*&
shared_namesequential/dense/bias
|
)sequential/dense/bias/Read/ReadVariableOpReadVariableOpsequential/dense/bias*
_output_shapes	
:?*
dtype0
?
sequential/dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??**
shared_namesequential/dense_1/kernel
?
-sequential/dense_1/kernel/Read/ReadVariableOpReadVariableOpsequential/dense_1/kernel* 
_output_shapes
:
??*
dtype0
?
sequential/dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*(
shared_namesequential/dense_1/bias
?
+sequential/dense_1/bias/Read/ReadVariableOpReadVariableOpsequential/dense_1/bias*
_output_shapes	
:?*
dtype0
?
sequential/dense_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??**
shared_namesequential/dense_2/kernel
?
-sequential/dense_2/kernel/Read/ReadVariableOpReadVariableOpsequential/dense_2/kernel* 
_output_shapes
:
??*
dtype0
?
sequential/dense_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*(
shared_namesequential/dense_2/bias
?
+sequential/dense_2/bias/Read/ReadVariableOpReadVariableOpsequential/dense_2/bias*
_output_shapes	
:?*
dtype0
?
sequential/dense_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?**
shared_namesequential/dense_3/kernel
?
-sequential/dense_3/kernel/Read/ReadVariableOpReadVariableOpsequential/dense_3/kernel*
_output_shapes
:	?*
dtype0
?
sequential/dense_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_namesequential/dense_3/bias

+sequential/dense_3/bias/Read/ReadVariableOpReadVariableOpsequential/dense_3/bias*
_output_shapes
:*
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
k

hash_tableHashTableV2*
_output_shapes
: *
	key_dtype0*
shared_name300*
value_dtype0	
m
hash_table_1HashTableV2*
_output_shapes
: *
	key_dtype0*
shared_name342*
value_dtype0	
m
hash_table_2HashTableV2*
_output_shapes
: *
	key_dtype0*
shared_name376*
value_dtype0	
m
hash_table_3HashTableV2*
_output_shapes
: *
	key_dtype0*
shared_name410*
value_dtype0	
m
hash_table_4HashTableV2*
_output_shapes
: *
	key_dtype0*
shared_name444*
value_dtype0	
m
hash_table_5HashTableV2*
_output_shapes
: *
	key_dtype0*
shared_name478*
value_dtype0	
m
hash_table_6HashTableV2*
_output_shapes
: *
	key_dtype0*
shared_name512*
value_dtype0	
m
hash_table_7HashTableV2*
_output_shapes
: *
	key_dtype0	*
shared_name548*
value_dtype0	
m
hash_table_8HashTableV2*
_output_shapes
: *
	key_dtype0	*
shared_name584*
value_dtype0	
m
hash_table_9HashTableV2*
_output_shapes
: *
	key_dtype0	*
shared_name636*
value_dtype0	
n
hash_table_10HashTableV2*
_output_shapes
: *
	key_dtype0	*
shared_name712*
value_dtype0	
n
hash_table_11HashTableV2*
_output_shapes
: *
	key_dtype0	*
shared_name748*
value_dtype0	
n
hash_table_12HashTableV2*
_output_shapes
: *
	key_dtype0	*
shared_name784*
value_dtype0	
n
hash_table_13HashTableV2*
_output_shapes
: *
	key_dtype0*
shared_name818*
value_dtype0	
n
hash_table_14HashTableV2*
_output_shapes
: *
	key_dtype0*
shared_name860*
value_dtype0	
n
hash_table_15HashTableV2*
_output_shapes
: *
	key_dtype0	*
shared_name904*
value_dtype0	
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
p
true_positivesVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametrue_positives
i
"true_positives/Read/ReadVariableOpReadVariableOptrue_positives*
_output_shapes
: *
dtype0
r
false_positivesVarHandleOp*
_output_shapes
: *
dtype0*
shape: * 
shared_namefalse_positives
k
#false_positives/Read/ReadVariableOpReadVariableOpfalse_positives*
_output_shapes
: *
dtype0
r
false_negativesVarHandleOp*
_output_shapes
: *
dtype0*
shape: * 
shared_namefalse_negatives
k
#false_negatives/Read/ReadVariableOpReadVariableOpfalse_negatives*
_output_shapes
: *
dtype0
|
weights_intermediateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_nameweights_intermediate
u
(weights_intermediate/Read/ReadVariableOpReadVariableOpweights_intermediate*
_output_shapes
: *
dtype0
?
Adam/sequential/dense/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*/
shared_name Adam/sequential/dense/kernel/m
?
2Adam/sequential/dense/kernel/m/Read/ReadVariableOpReadVariableOpAdam/sequential/dense/kernel/m* 
_output_shapes
:
??*
dtype0
?
Adam/sequential/dense/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*-
shared_nameAdam/sequential/dense/bias/m
?
0Adam/sequential/dense/bias/m/Read/ReadVariableOpReadVariableOpAdam/sequential/dense/bias/m*
_output_shapes	
:?*
dtype0
?
 Adam/sequential/dense_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*1
shared_name" Adam/sequential/dense_1/kernel/m
?
4Adam/sequential/dense_1/kernel/m/Read/ReadVariableOpReadVariableOp Adam/sequential/dense_1/kernel/m* 
_output_shapes
:
??*
dtype0
?
Adam/sequential/dense_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*/
shared_name Adam/sequential/dense_1/bias/m
?
2Adam/sequential/dense_1/bias/m/Read/ReadVariableOpReadVariableOpAdam/sequential/dense_1/bias/m*
_output_shapes	
:?*
dtype0
?
 Adam/sequential/dense_2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*1
shared_name" Adam/sequential/dense_2/kernel/m
?
4Adam/sequential/dense_2/kernel/m/Read/ReadVariableOpReadVariableOp Adam/sequential/dense_2/kernel/m* 
_output_shapes
:
??*
dtype0
?
Adam/sequential/dense_2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*/
shared_name Adam/sequential/dense_2/bias/m
?
2Adam/sequential/dense_2/bias/m/Read/ReadVariableOpReadVariableOpAdam/sequential/dense_2/bias/m*
_output_shapes	
:?*
dtype0
?
 Adam/sequential/dense_3/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*1
shared_name" Adam/sequential/dense_3/kernel/m
?
4Adam/sequential/dense_3/kernel/m/Read/ReadVariableOpReadVariableOp Adam/sequential/dense_3/kernel/m*
_output_shapes
:	?*
dtype0
?
Adam/sequential/dense_3/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*/
shared_name Adam/sequential/dense_3/bias/m
?
2Adam/sequential/dense_3/bias/m/Read/ReadVariableOpReadVariableOpAdam/sequential/dense_3/bias/m*
_output_shapes
:*
dtype0
?
Adam/sequential/dense/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*/
shared_name Adam/sequential/dense/kernel/v
?
2Adam/sequential/dense/kernel/v/Read/ReadVariableOpReadVariableOpAdam/sequential/dense/kernel/v* 
_output_shapes
:
??*
dtype0
?
Adam/sequential/dense/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*-
shared_nameAdam/sequential/dense/bias/v
?
0Adam/sequential/dense/bias/v/Read/ReadVariableOpReadVariableOpAdam/sequential/dense/bias/v*
_output_shapes	
:?*
dtype0
?
 Adam/sequential/dense_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*1
shared_name" Adam/sequential/dense_1/kernel/v
?
4Adam/sequential/dense_1/kernel/v/Read/ReadVariableOpReadVariableOp Adam/sequential/dense_1/kernel/v* 
_output_shapes
:
??*
dtype0
?
Adam/sequential/dense_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*/
shared_name Adam/sequential/dense_1/bias/v
?
2Adam/sequential/dense_1/bias/v/Read/ReadVariableOpReadVariableOpAdam/sequential/dense_1/bias/v*
_output_shapes	
:?*
dtype0
?
 Adam/sequential/dense_2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*1
shared_name" Adam/sequential/dense_2/kernel/v
?
4Adam/sequential/dense_2/kernel/v/Read/ReadVariableOpReadVariableOp Adam/sequential/dense_2/kernel/v* 
_output_shapes
:
??*
dtype0
?
Adam/sequential/dense_2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*/
shared_name Adam/sequential/dense_2/bias/v
?
2Adam/sequential/dense_2/bias/v/Read/ReadVariableOpReadVariableOpAdam/sequential/dense_2/bias/v*
_output_shapes	
:?*
dtype0
?
 Adam/sequential/dense_3/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*1
shared_name" Adam/sequential/dense_3/kernel/v
?
4Adam/sequential/dense_3/kernel/v/Read/ReadVariableOpReadVariableOp Adam/sequential/dense_3/kernel/v*
_output_shapes
:	?*
dtype0
?
Adam/sequential/dense_3/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*/
shared_name Adam/sequential/dense_3/bias/v
?
2Adam/sequential/dense_3/bias/v/Read/ReadVariableOpReadVariableOpAdam/sequential/dense_3/bias/v*
_output_shapes
:*
dtype0
P
ConstConst*
_output_shapes
: *
dtype0	*
valueB	 R
?????????
R
Const_1Const*
_output_shapes
: *
dtype0	*
valueB	 R
?????????
R
Const_2Const*
_output_shapes
: *
dtype0	*
valueB	 R
?????????
R
Const_3Const*
_output_shapes
: *
dtype0	*
valueB	 R
?????????
R
Const_4Const*
_output_shapes
: *
dtype0	*
valueB	 R
?????????
R
Const_5Const*
_output_shapes
: *
dtype0	*
valueB	 R
?????????
R
Const_6Const*
_output_shapes
: *
dtype0	*
valueB	 R
?????????
R
Const_7Const*
_output_shapes
: *
dtype0	*
valueB	 R
?????????
R
Const_8Const*
_output_shapes
: *
dtype0	*
valueB	 R
?????????
R
Const_9Const*
_output_shapes
: *
dtype0	*
valueB	 R
?????????
S
Const_10Const*
_output_shapes
: *
dtype0	*
valueB	 R
?????????
S
Const_11Const*
_output_shapes
: *
dtype0	*
valueB	 R
?????????
S
Const_12Const*
_output_shapes
: *
dtype0	*
valueB	 R
?????????
S
Const_13Const*
_output_shapes
: *
dtype0	*
valueB	 R
?????????
S
Const_14Const*
_output_shapes
: *
dtype0	*
valueB	 R
?????????
S
Const_15Const*
_output_shapes
: *
dtype0	*
valueB	 R
?????????
?
Const_16Const*
_output_shapes
:!*
dtype0*?
value?B?!BX99BX04BY99BX05BX01BX10BX02BY16BX09BX12BY09BY17BX08BY12BY05BY03BY15BY01BX11BY02BY19BX07BY18BY11BX03BY04BY10BY08BX06BY06BY14BY13BY07
?
Const_17Const*
_output_shapes
:!*
dtype0	*?
value?B?	!"?                                                                	       
                                                                                                                                                                  
?
Const_18Const*
_output_shapes
:"*
dtype0*?
value?B?"BNoneBX99BX05BX10BX04BY10BX09BY09BY16BX03BX12BX01BX08BY05BY04BY14BX02BY08BY06BY11BY07BY03BY99BY01BY13BY15BX11BY18BY17BX07BX06BY02BY12BY19
?
Const_19Const*
_output_shapes
:"*
dtype0	*?
value?B?	""?                                                                	       
                                                                                                                                                                  !       
U
Const_20Const*
_output_shapes
:*
dtype0*
valueBBNBY
a
Const_21Const*
_output_shapes
:*
dtype0	*%
valueB	"               
U
Const_22Const*
_output_shapes
:*
dtype0*
valueBBNBY
a
Const_23Const*
_output_shapes
:*
dtype0	*%
valueB	"               
U
Const_24Const*
_output_shapes
:*
dtype0*
valueBBNBY
a
Const_25Const*
_output_shapes
:*
dtype0	*%
valueB	"               
U
Const_26Const*
_output_shapes
:*
dtype0*
valueBBNBY
a
Const_27Const*
_output_shapes
:*
dtype0	*%
valueB	"               
U
Const_28Const*
_output_shapes
:*
dtype0*
valueBBNBY
a
Const_29Const*
_output_shapes
:*
dtype0	*%
valueB	"               
?
Const_30Const*
_output_shapes
:*
dtype0	*}
valuetBr	"h                                                                                    	       
?
Const_31Const*
_output_shapes
:*
dtype0	*}
valuetBr	"h                                                                	       
                     
?
Const_32Const*
_output_shapes
:<*
dtype0	*?
value?B?	<"??      ?       ?       ?      ?       ?      ?      ?       ?      p       ?      ?       ?       ?       ?       b      ?       ?       ?       _      F      C      A      ?       `      X      ?       W      D      ?      ?       E      U      ?       ?      ?      o       ?       ?       V      ?      ?       ?       B      ?      L      K      7      a      ?       ?       ?       G      ?      ?      ?       ?       8      ?      9      
?
Const_33Const*
_output_shapes
:<*
dtype0	*?
value?B?	<"?                                                                	       
                                                                                                                                                                  !       "       #       $       %       &       '       (       )       *       +       ,       -       .       /       0       1       2       3       4       5       6       7       8       9       :       ;       
a
Const_34Const*
_output_shapes
:*
dtype0	*%
valueB	"              
a
Const_35Const*
_output_shapes
:*
dtype0	*%
valueB	"               
?
Const_36Const*
_output_shapes
:*
dtype0	*?
value?B?	"?`?      ?      ?N      ?u      t'      LO      |?      ??      )      <(      ?N      ?(      ?v      ?      ?'      ?O      ??      l?      ?u      \v      P      
?
Const_37Const*
_output_shapes
:*
dtype0	*?
value?B?	"?                                                                	       
                                                                             
?
Const_38Const*
_output_shapes
:*
dtype0	*U
valueLBJ	"@              c                                          
?
Const_39Const*
_output_shapes
:*
dtype0	*U
valueLBJ	"@                                                         
q
Const_40Const*
_output_shapes
:*
dtype0	*5
value,B*	"                             
q
Const_41Const*
_output_shapes
:*
dtype0	*5
value,B*	"                              
?
Const_42Const*
_output_shapes	
:?*
dtype0*?
value?B??BOC03BSB12BZZBEG99BEG04BEG07BEE13BND99BND01BEF99BLC14BEG06BEH14BEA01BEH10BLC08BEF06BEG02BEG01BLA02BLA01BSA01BLA05BLA07BNA09BEE99BNC03BHE04BEH01BLA11BEA04BEE01BSI07BNA08BSB08BLA99BSG03BEE10BLC01BLB04BNB04BNC04BNC05BNC99BEA09BEB07BLC99BEE03BED99BLC02BEE02BLC04BED03BNB09BEI99BEI11BEA02BEC11BND07BEI09BSB06BEH02BEB04BLB16BLC15BEA10BED10BNC10BSC08BOC99BED08BSB99BSD02BED07BLC13BLC03BSA99BEC99BEI08BEB01BHE14BEH11BLB12BNA04BSB09BLB17BSG02BLB13BEI05BNB05BEA99BEA14BEE12BND12BEE06BLA03BEG09BNC02BEA11BED04BEA06BEB03BEE04BEC02BEF04BLC10BEC01BEA12BEB02BNC07BLC06BEH05BNC01BNA02BLA04BSC09BNA99BNB06BNB01BLB02BEI04BEH03BLA06BED01BEA13BOA01BEE11BLB18BLA09BEE08BEG05BEA07BNC09BEB99BOA04BEH09BOA02BED11BEH04BEI03BOB01BED05BNC08BLB09BEC04BEI12BEA05BLC07BLB07BND11BSE06BNA05BLB03BNA06BEG03BLB01BLA08BLC05BEE14BND06BNC06BHE06BEG10BEA08BEC05BEC03BEI02BLC09BEE09BNA03BEE07BEH15BND10BSC12BND04BND13BEH06BSE05BND03BEA03BNA01BND08BEB08BEG08BEB06BED06BNB02BEF05BNA07BZZ99BEH07BEB05BEE05BNB07BLB05BED02BLA10BEA15BLB10BEC07BLB06BSI05BEF01BEC08BND14BNB03BED09BND02BSC10BLC12BHC06BEH99BSC11BEI06BEI10BNA10BEH13BOB99BEI01BSA07BLB15BEF03BSH01BOA99BSC07BNB08BOC04BOB02BOA03BSG07BSF01BSH02BND05BSC13BEF02BSH06BOC01BHE13BLB11BHE03BLB14BSH07BNB99BSC01BHB04BHC01BEH08BLB08BSG05BEC06BSF02BSB11BEC09BSD07BSB07BLC11BSI01BEH12BSH99BSC99BSC05BHE15BND09BHE09BLB19BLB20BLB99BSI99BSI02BSI03BHA01BHD08BSD03BSD04BHC02BHD02BHD10BHB08BHA03BSF07BSD99BHD06BSC16BHA05BSH04BHE10BHA02BSA04BHD03BSC02BSD08BHE01BHD07BSG06BHA04BSB10BSC06BSA02BHD99BSA05BSC14BSD05BEI07BHE12BHE08BHC05BSA06BSG01BSC03BHC03BHE02BSF04BSD09BSB01BHA06BSE04BHB03BSF03BSG04BHE05BHB07BHD04BHA99BHC09BHC07BSB02BHB05BHC08BHE99BSC15BHB01BSH08BHD09BSI04BHD05BSC04BSA03BSE03BHE11BHC12BHD01BSD06BHE07BHC99BSB05BHB09BSD01BHB10BSE99BHC04BSF05BSF99BHB06BSE02BHB02BSH03BHA07BEC10BSG99BOC02BSF06BSE01BSI06
?
Const_43Const*
_output_shapes	
:?*
dtype0	*?
value?B?	?"?                                                                	       
                                                                                                                                                                  !       "       #       $       %       &       '       (       )       *       +       ,       -       .       /       0       1       2       3       4       5       6       7       8       9       :       ;       <       =       >       ?       @       A       B       C       D       E       F       G       H       I       J       K       L       M       N       O       P       Q       R       S       T       U       V       W       X       Y       Z       [       \       ]       ^       _       `       a       b       c       d       e       f       g       h       i       j       k       l       m       n       o       p       q       r       s       t       u       v       w       x       y       z       {       |       }       ~              ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?                                                              	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      
?
Const_44Const*
_output_shapes	
:?*
dtype0*?
value?B??BNoneBEE10BEE02BEG07BEG10BEF99BEI99BEG99BEG04BOC03BND02BEA14BEH13BEG02BEG06BLC14BLA01BLA02BLA99BLC02BLB12BHE99BEB07BEE11BSI07BEE01BHE14BSA01BNC09BEF06BNC04BLC01BLC04BNB09BEC11BEE13BSB11BHE04BND07BEI11BND05BEH08BLC13BNC06BSI02BEH11BOC01BEA05BLC03BEC02BSI99BNC01BEA02BNC10BNA08BLA04BEI01BED01BEA09BLA06BLB03BNB05BEI05BND06BEE07BLB18BNC02BEA07BEA04BNB06BNA07BNC08BEA15BED05BEI12BEC03BLB07BED03BLA07BLA08BED11BEE03BED04BND13BLA03BNA01BEG01BLC10BEB01BEI04BEH02BNB01BLA10BEE14BEA08BED10BEB03BOB01BEF05BNA05BLB05BEG09BEB02BOA01BLC07BEE06BND03BSF01BLC08BEC04BNB08BEH03BNC03BNB04BEE99BEG08BEE09BNA09BEC01BEE05BNC05BHE08BOA04BEC05BEA06BEA01BEG03BSC09BND14BLA05BND08BLC05BEE04BEB99BSI05BOA02BEE12BEA10BEF04BEA11BEI02BEH07BNA06BLC15BOA03BLB17BNB07BED08BNB02BED07BED99BLA09BEH06BED02BEB08BEH01BEH09BEI03BND01BNA02BEA12BED09BLC12BEA13BHC02BEH05BEB05BLB13BSD01BEF01BEB04BEC06BEH12BEE08BEC99BEB06BSC11BEI10BOB02BLB06BLB04BND10BLB01BSH01BHA06BSF02BEH10BNC07BSE06BSD04BNA10BLC06BEC09BEF03BEF02BSB12BSA07BNA04BHA05BND12BSC10BHE10BOC99BSG07BEI06BSC12BEA03BLA11BSD08BSG04BLC11BNA03BSH03BSF07BEI09BSF03BOA99BSB02BLB16BLB10BOB99BSG02BSH07BSC07BED06BLC09BSC02BHE05BNB03BHE13BLB99BSG05BND04BSD99BSI01BLB14BEI08BLB09BNB99BEH04BSH02BHE15BND11BSE03BEA99BLC99BHE12BSH06BHE11BLB11BSC06BND09BEH14BSI03BLB15BHA01BSH08BSC13BNC99BEH15BSH99BLB02BSF99BEH99BHC01BSC04BHA02BSC16BEC10BHA99BSC99BSD03BSG03BSD02BEG05BHE06BEC07BSD07BSC01BHC05BHB02BSF05BHB05BSC08BHB03BHD02BHB01BHB08BHD03BSD09BSD05BHB04BSH04BSB06BSC15BHE02BHB10BSA06BHE09BSD06BHD99BNA99BHE01BHA04BHD06BSC03BHE03BHD05BSB05BSG99BSE05BSG06BEC08BHE07BLB20BSC05BOC02BHA07BSG01BHA03BSC14BHD01BSA05BSB10BSB01BLB08BHC03BLB19BSE99BOC04BSE04BND99BEI07BHC12BSB08BSF04BSI04BSA99
?
Const_45Const*
_output_shapes	
:?*
dtype0	*?
value?B?	?"?                                                                	       
                                                                                                                                                                  !       "       #       $       %       &       '       (       )       *       +       ,       -       .       /       0       1       2       3       4       5       6       7       8       9       :       ;       <       =       >       ?       @       A       B       C       D       E       F       G       H       I       J       K       L       M       N       O       P       Q       R       S       T       U       V       W       X       Y       Z       [       \       ]       ^       _       `       a       b       c       d       e       f       g       h       i       j       k       l       m       n       o       p       q       r       s       t       u       v       w       x       y       z       {       |       }       ~              ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?                                                              	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      
?
Const_46Const*
_output_shapes
:*
dtype0	*?
value?B?	"?p     <(      ?(      ??      ?N      ?      ?N      (?      ??      ?u      ??      |?      LO      ??      t'      ??      ?v      ?'      l?      ?u      ?      Н      \v      
?
Const_47Const*
_output_shapes
:*
dtype0	*?
value?B?	"?                                                                	       
                                                                                           
?
StatefulPartitionedCallStatefulPartitionedCall
hash_tableConst_16Const_17*
Tin
2	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *$
fR
__inference_<lambda>_133946
?
StatefulPartitionedCall_1StatefulPartitionedCallhash_table_1Const_18Const_19*
Tin
2	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *$
fR
__inference_<lambda>_133954
?
StatefulPartitionedCall_2StatefulPartitionedCallhash_table_2Const_20Const_21*
Tin
2	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *$
fR
__inference_<lambda>_133962
?
StatefulPartitionedCall_3StatefulPartitionedCallhash_table_3Const_22Const_23*
Tin
2	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *$
fR
__inference_<lambda>_133970
?
StatefulPartitionedCall_4StatefulPartitionedCallhash_table_4Const_24Const_25*
Tin
2	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *$
fR
__inference_<lambda>_133978
?
StatefulPartitionedCall_5StatefulPartitionedCallhash_table_5Const_26Const_27*
Tin
2	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *$
fR
__inference_<lambda>_133986
?
StatefulPartitionedCall_6StatefulPartitionedCallhash_table_6Const_28Const_29*
Tin
2	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *$
fR
__inference_<lambda>_133994
?
StatefulPartitionedCall_7StatefulPartitionedCallhash_table_7Const_30Const_31*
Tin
2		*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *$
fR
__inference_<lambda>_134002
?
StatefulPartitionedCall_8StatefulPartitionedCallhash_table_8Const_32Const_33*
Tin
2		*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *$
fR
__inference_<lambda>_134010
?
StatefulPartitionedCall_9StatefulPartitionedCallhash_table_9Const_34Const_35*
Tin
2		*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *$
fR
__inference_<lambda>_134018
?
StatefulPartitionedCall_10StatefulPartitionedCallhash_table_10Const_36Const_37*
Tin
2		*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *$
fR
__inference_<lambda>_134026
?
StatefulPartitionedCall_11StatefulPartitionedCallhash_table_11Const_38Const_39*
Tin
2		*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *$
fR
__inference_<lambda>_134034
?
StatefulPartitionedCall_12StatefulPartitionedCallhash_table_12Const_40Const_41*
Tin
2		*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *$
fR
__inference_<lambda>_134042
?
StatefulPartitionedCall_13StatefulPartitionedCallhash_table_13Const_42Const_43*
Tin
2	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *$
fR
__inference_<lambda>_134050
?
StatefulPartitionedCall_14StatefulPartitionedCallhash_table_14Const_44Const_45*
Tin
2	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *$
fR
__inference_<lambda>_134058
?
StatefulPartitionedCall_15StatefulPartitionedCallhash_table_15Const_46Const_47*
Tin
2		*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *$
fR
__inference_<lambda>_134066
?
NoOpNoOp^StatefulPartitionedCall^StatefulPartitionedCall_1^StatefulPartitionedCall_10^StatefulPartitionedCall_11^StatefulPartitionedCall_12^StatefulPartitionedCall_13^StatefulPartitionedCall_14^StatefulPartitionedCall_15^StatefulPartitionedCall_2^StatefulPartitionedCall_3^StatefulPartitionedCall_4^StatefulPartitionedCall_5^StatefulPartitionedCall_6^StatefulPartitionedCall_7^StatefulPartitionedCall_8^StatefulPartitionedCall_9
?E
Const_48Const"/device:CPU:0*
_output_shapes
: *
dtype0*?D
value?DB?D B?D
?
layer-0
layer_with_weights-0
layer-1
layer-2
layer_with_weights-1
layer-3
layer-4
layer_with_weights-2
layer-5
layer-6
layer_with_weights-3
layer-7
		optimizer

_build_input_shape
	variables
trainable_variables
regularization_losses
	keras_api

signatures
x
_feature_columns

_resources
	variables
trainable_variables
regularization_losses
	keras_api
h

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
R
	variables
trainable_variables
regularization_losses
	keras_api
h

 kernel
!bias
"	variables
#trainable_variables
$regularization_losses
%	keras_api
R
&	variables
'trainable_variables
(regularization_losses
)	keras_api
h

*kernel
+bias
,	variables
-trainable_variables
.regularization_losses
/	keras_api
R
0	variables
1trainable_variables
2regularization_losses
3	keras_api
h

4kernel
5bias
6	variables
7trainable_variables
8regularization_losses
9	keras_api
?
:iter

;beta_1

<beta_2
	=decay
>learning_ratem?m? m?!m?*m?+m?4m?5m?v?v? v?!v?*v?+v?4v?5v?
 
8
0
1
 2
!3
*4
+5
46
57
8
0
1
 2
!3
*4
+5
46
57
 
?
?non_trainable_variables

@layers
Ametrics
Blayer_regularization_losses
Clayer_metrics
	variables
trainable_variables
regularization_losses
 
 
?
DApplication_Area_1
EApplication_Area_2
FCowork_Abroad
G
Cowork_Cor
HCowork_Inst
I
Cowork_Uni
J
Cowork_etc
KEcon_Social
L
Green_Tech
M
Multi_Year
NNational_Strategy_2
ORnD_Org
P	RnD_Stage
QSTP_Code_11
RSTP_Code_21

SSixT_2
 
 
 
?
Tnon_trainable_variables

Ulayers
Vmetrics
Wlayer_regularization_losses
Xlayer_metrics
	variables
trainable_variables
regularization_losses
ca
VARIABLE_VALUEsequential/dense/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
_]
VARIABLE_VALUEsequential/dense/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
?
Ynon_trainable_variables

Zlayers
[metrics
\layer_regularization_losses
]layer_metrics
	variables
trainable_variables
regularization_losses
 
 
 
?
^non_trainable_variables

_layers
`metrics
alayer_regularization_losses
blayer_metrics
	variables
trainable_variables
regularization_losses
ec
VARIABLE_VALUEsequential/dense_1/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
a_
VARIABLE_VALUEsequential/dense_1/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE

 0
!1

 0
!1
 
?
cnon_trainable_variables

dlayers
emetrics
flayer_regularization_losses
glayer_metrics
"	variables
#trainable_variables
$regularization_losses
 
 
 
?
hnon_trainable_variables

ilayers
jmetrics
klayer_regularization_losses
llayer_metrics
&	variables
'trainable_variables
(regularization_losses
ec
VARIABLE_VALUEsequential/dense_2/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
a_
VARIABLE_VALUEsequential/dense_2/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE

*0
+1

*0
+1
 
?
mnon_trainable_variables

nlayers
ometrics
player_regularization_losses
qlayer_metrics
,	variables
-trainable_variables
.regularization_losses
 
 
 
?
rnon_trainable_variables

slayers
tmetrics
ulayer_regularization_losses
vlayer_metrics
0	variables
1trainable_variables
2regularization_losses
ec
VARIABLE_VALUEsequential/dense_3/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
a_
VARIABLE_VALUEsequential/dense_3/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE

40
51

40
51
 
?
wnon_trainable_variables

xlayers
ymetrics
zlayer_regularization_losses
{layer_metrics
6	variables
7trainable_variables
8regularization_losses
HF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
 
8
0
1
2
3
4
5
6
7

|0
}1
 
 

~Application_Area_1_lookup

Application_Area_2_lookup

?Cowork_Abroad_lookup

?Cowork_Cor_lookup

?Cowork_Inst_lookup

?Cowork_Uni_lookup

?Cowork_etc_lookup

?Econ_Social_lookup

?Green_Tech_lookup

?Multi_Year_lookup
!
?National_Strategy_2_lookup

?RnD_Org_lookup

?RnD_Stage_lookup

?STP_Code_11_lookup

?STP_Code_21_lookup

?SixT_2_lookup
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
8

?total

?count
?	variables
?	keras_api
?
?
init_shape
?true_positives
?false_positives
?false_negatives
?weights_intermediate
?	variables
?	keras_api

?_initializer

?_initializer

?_initializer

?_initializer

?_initializer

?_initializer

?_initializer

?_initializer

?_initializer

?_initializer

?_initializer

?_initializer

?_initializer

?_initializer

?_initializer

?_initializer
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

?0
?1

?	variables
 
a_
VARIABLE_VALUEtrue_positives=keras_api/metrics/1/true_positives/.ATTRIBUTES/VARIABLE_VALUE
ca
VARIABLE_VALUEfalse_positives>keras_api/metrics/1/false_positives/.ATTRIBUTES/VARIABLE_VALUE
ca
VARIABLE_VALUEfalse_negatives>keras_api/metrics/1/false_negatives/.ATTRIBUTES/VARIABLE_VALUE
mk
VARIABLE_VALUEweights_intermediateCkeras_api/metrics/1/weights_intermediate/.ATTRIBUTES/VARIABLE_VALUE
 
?0
?1
?2
?3

?	variables
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
??
VARIABLE_VALUEAdam/sequential/dense/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdam/sequential/dense/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE Adam/sequential/dense_1/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdam/sequential/dense_1/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE Adam/sequential/dense_2/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdam/sequential/dense_2/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE Adam/sequential/dense_3/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdam/sequential/dense_3/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdam/sequential/dense/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdam/sequential/dense/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE Adam/sequential/dense_1/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdam/sequential/dense_1/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE Adam/sequential/dense_2/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdam/sequential/dense_2/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE Adam/sequential/dense_3/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdam/sequential/dense_3/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?
"serving_default_Application_Area_1Placeholder*'
_output_shapes
:?????????*
dtype0*
shape:?????????
?
)serving_default_Application_Area_1_WeightPlaceholder*'
_output_shapes
:?????????*
dtype0*
shape:?????????
?
"serving_default_Application_Area_2Placeholder*'
_output_shapes
:?????????*
dtype0*
shape:?????????
?
)serving_default_Application_Area_2_WeightPlaceholder*'
_output_shapes
:?????????*
dtype0*
shape:?????????
?
serving_default_Cowork_AbroadPlaceholder*'
_output_shapes
:?????????*
dtype0*
shape:?????????
}
serving_default_Cowork_CorPlaceholder*'
_output_shapes
:?????????*
dtype0*
shape:?????????
~
serving_default_Cowork_InstPlaceholder*'
_output_shapes
:?????????*
dtype0*
shape:?????????
}
serving_default_Cowork_UniPlaceholder*'
_output_shapes
:?????????*
dtype0*
shape:?????????
}
serving_default_Cowork_etcPlaceholder*'
_output_shapes
:?????????*
dtype0*
shape:?????????
~
serving_default_Econ_SocialPlaceholder*'
_output_shapes
:?????????*
dtype0	*
shape:?????????
}
serving_default_Green_TechPlaceholder*'
_output_shapes
:?????????*
dtype0	*
shape:?????????

serving_default_Log_DurationPlaceholder*'
_output_shapes
:?????????*
dtype0*
shape:?????????

serving_default_Log_RnD_FundPlaceholder*'
_output_shapes
:?????????*
dtype0*
shape:?????????
}
serving_default_Multi_YearPlaceholder*'
_output_shapes
:?????????*
dtype0	*
shape:?????????

serving_default_N_Patent_AppPlaceholder*'
_output_shapes
:?????????*
dtype0*
shape:?????????

serving_default_N_Patent_RegPlaceholder*'
_output_shapes
:?????????*
dtype0*
shape:?????????
?
"serving_default_N_of_Korean_PatentPlaceholder*'
_output_shapes
:?????????*
dtype0*
shape:?????????
}
serving_default_N_of_PaperPlaceholder*'
_output_shapes
:?????????*
dtype0*
shape:?????????
{
serving_default_N_of_SCIPlaceholder*'
_output_shapes
:?????????*
dtype0*
shape:?????????
?
#serving_default_National_Strategy_2Placeholder*'
_output_shapes
:?????????*
dtype0	*
shape:?????????
z
serving_default_RnD_OrgPlaceholder*'
_output_shapes
:?????????*
dtype0	*
shape:?????????
|
serving_default_RnD_StagePlaceholder*'
_output_shapes
:?????????*
dtype0	*
shape:?????????
~
serving_default_STP_Code_11Placeholder*'
_output_shapes
:?????????*
dtype0*
shape:?????????
?
!serving_default_STP_Code_1_WeightPlaceholder*'
_output_shapes
:?????????*
dtype0*
shape:?????????
~
serving_default_STP_Code_21Placeholder*'
_output_shapes
:?????????*
dtype0*
shape:?????????
?
!serving_default_STP_Code_2_WeightPlaceholder*'
_output_shapes
:?????????*
dtype0*
shape:?????????
y
serving_default_SixT_2Placeholder*'
_output_shapes
:?????????*
dtype0	*
shape:?????????
w
serving_default_YearPlaceholder*'
_output_shapes
:?????????*
dtype0*
shape:?????????
?
StatefulPartitionedCall_16StatefulPartitionedCall"serving_default_Application_Area_1)serving_default_Application_Area_1_Weight"serving_default_Application_Area_2)serving_default_Application_Area_2_Weightserving_default_Cowork_Abroadserving_default_Cowork_Corserving_default_Cowork_Instserving_default_Cowork_Uniserving_default_Cowork_etcserving_default_Econ_Socialserving_default_Green_Techserving_default_Log_Durationserving_default_Log_RnD_Fundserving_default_Multi_Yearserving_default_N_Patent_Appserving_default_N_Patent_Reg"serving_default_N_of_Korean_Patentserving_default_N_of_Paperserving_default_N_of_SCI#serving_default_National_Strategy_2serving_default_RnD_Orgserving_default_RnD_Stageserving_default_STP_Code_11!serving_default_STP_Code_1_Weightserving_default_STP_Code_21!serving_default_STP_Code_2_Weightserving_default_SixT_2serving_default_Year
hash_tableConsthash_table_1Const_1hash_table_2Const_2hash_table_3Const_3hash_table_4Const_4hash_table_5Const_5hash_table_6Const_6hash_table_7Const_7hash_table_8Const_8hash_table_9Const_9hash_table_10Const_10hash_table_11Const_11hash_table_12Const_12hash_table_13Const_13hash_table_14Const_14hash_table_15Const_15sequential/dense/kernelsequential/dense/biassequential/dense_1/kernelsequential/dense_1/biassequential/dense_2/kernelsequential/dense_2/biassequential/dense_3/kernelsequential/dense_3/bias*O
TinH
F2D																							*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????**
_read_only_resource_inputs

<=>?@ABC*0
config_proto 

CPU

GPU2*0J 8? *-
f(R&
$__inference_signature_wrapper_130910
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_17StatefulPartitionedCallsaver_filename+sequential/dense/kernel/Read/ReadVariableOp)sequential/dense/bias/Read/ReadVariableOp-sequential/dense_1/kernel/Read/ReadVariableOp+sequential/dense_1/bias/Read/ReadVariableOp-sequential/dense_2/kernel/Read/ReadVariableOp+sequential/dense_2/bias/Read/ReadVariableOp-sequential/dense_3/kernel/Read/ReadVariableOp+sequential/dense_3/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp"true_positives/Read/ReadVariableOp#false_positives/Read/ReadVariableOp#false_negatives/Read/ReadVariableOp(weights_intermediate/Read/ReadVariableOp2Adam/sequential/dense/kernel/m/Read/ReadVariableOp0Adam/sequential/dense/bias/m/Read/ReadVariableOp4Adam/sequential/dense_1/kernel/m/Read/ReadVariableOp2Adam/sequential/dense_1/bias/m/Read/ReadVariableOp4Adam/sequential/dense_2/kernel/m/Read/ReadVariableOp2Adam/sequential/dense_2/bias/m/Read/ReadVariableOp4Adam/sequential/dense_3/kernel/m/Read/ReadVariableOp2Adam/sequential/dense_3/bias/m/Read/ReadVariableOp2Adam/sequential/dense/kernel/v/Read/ReadVariableOp0Adam/sequential/dense/bias/v/Read/ReadVariableOp4Adam/sequential/dense_1/kernel/v/Read/ReadVariableOp2Adam/sequential/dense_1/bias/v/Read/ReadVariableOp4Adam/sequential/dense_2/kernel/v/Read/ReadVariableOp2Adam/sequential/dense_2/bias/v/Read/ReadVariableOp4Adam/sequential/dense_3/kernel/v/Read/ReadVariableOp2Adam/sequential/dense_3/bias/v/Read/ReadVariableOpConst_48*0
Tin)
'2%	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *(
f#R!
__inference__traced_save_134301
?	
StatefulPartitionedCall_18StatefulPartitionedCallsaver_filenamesequential/dense/kernelsequential/dense/biassequential/dense_1/kernelsequential/dense_1/biassequential/dense_2/kernelsequential/dense_2/biassequential/dense_3/kernelsequential/dense_3/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcounttrue_positivesfalse_positivesfalse_negativesweights_intermediateAdam/sequential/dense/kernel/mAdam/sequential/dense/bias/m Adam/sequential/dense_1/kernel/mAdam/sequential/dense_1/bias/m Adam/sequential/dense_2/kernel/mAdam/sequential/dense_2/bias/m Adam/sequential/dense_3/kernel/mAdam/sequential/dense_3/bias/mAdam/sequential/dense/kernel/vAdam/sequential/dense/bias/v Adam/sequential/dense_1/kernel/vAdam/sequential/dense_1/bias/v Adam/sequential/dense_2/kernel/vAdam/sequential/dense_2/bias/v Adam/sequential/dense_3/kernel/vAdam/sequential/dense_3/bias/v*/
Tin(
&2$*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *+
f&R$
"__inference__traced_restore_134416??5
?
?
&__inference_dense_layer_call_fn_133498

inputs
unknown:
??
	unknown_0:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_129113p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:??????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
__inference_<lambda>_1340582
.table_init859_lookuptableimportv2_table_handle*
&table_init859_lookuptableimportv2_keys,
(table_init859_lookuptableimportv2_values	
identity??!table_init859/LookupTableImportV2?
!table_init859/LookupTableImportV2LookupTableImportV2.table_init859_lookuptableimportv2_table_handle&table_init859_lookuptableimportv2_keys(table_init859_lookuptableimportv2_values*	
Tin0*

Tout0	*
_output_shapes
 J
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??L
IdentityIdentityConst:output:0^NoOp*
T0*
_output_shapes
: j
NoOpNoOp"^table_init859/LookupTableImportV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*#
_input_shapes
: :?:?2F
!table_init859/LookupTableImportV2!table_init859/LookupTableImportV2:!

_output_shapes	
:?:!

_output_shapes	
:?
?
;
__inference__creator_133727
identity??
hash_tablek

hash_tableHashTableV2*
_output_shapes
: *
	key_dtype0*
shared_name444*
value_dtype0	W
IdentityIdentityhash_table:table_handle:0^NoOp*
T0*
_output_shapes
: S
NoOpNoOp^hash_table*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 2

hash_table
hash_table
?
?
__inference_<lambda>_1339702
.table_init409_lookuptableimportv2_table_handle*
&table_init409_lookuptableimportv2_keys,
(table_init409_lookuptableimportv2_values	
identity??!table_init409/LookupTableImportV2?
!table_init409/LookupTableImportV2LookupTableImportV2.table_init409_lookuptableimportv2_table_handle&table_init409_lookuptableimportv2_keys(table_init409_lookuptableimportv2_values*	
Tin0*

Tout0	*
_output_shapes
 J
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??L
IdentityIdentityConst:output:0^NoOp*
T0*
_output_shapes
: j
NoOpNoOp"^table_init409/LookupTableImportV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*!
_input_shapes
: ::2F
!table_init409/LookupTableImportV2!table_init409/LookupTableImportV2: 

_output_shapes
:: 

_output_shapes
:
?
c
E__inference_dropout_1_layer_call_and_return_conditional_losses_133571

inputs

identity_1O
IdentityIdentityinputs*
T0*(
_output_shapes
:??????????\

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:??????????"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?6
?

/__inference_dense_features_layer_call_fn_132353
features_application_area_1&
"features_application_area_1_weight
features_application_area_2&
"features_application_area_2_weight
features_cowork_abroad
features_cowork_cor
features_cowork_inst
features_cowork_uni
features_cowork_etc
features_econ_social	
features_green_tech	
features_log_duration
features_log_rnd_fund
features_multi_year	
features_n_patent_app
features_n_patent_reg
features_n_of_korean_patent
features_n_of_paper
features_n_of_sci 
features_national_strategy_2	
features_rnd_org	
features_rnd_stage	
features_stp_code_11
features_stp_code_1_weight
features_stp_code_21
features_stp_code_2_weight
features_sixt_2	
features_year
unknown
	unknown_0	
	unknown_1
	unknown_2	
	unknown_3
	unknown_4	
	unknown_5
	unknown_6	
	unknown_7
	unknown_8	
	unknown_9

unknown_10	

unknown_11

unknown_12	

unknown_13

unknown_14	

unknown_15

unknown_16	

unknown_17

unknown_18	

unknown_19

unknown_20	

unknown_21

unknown_22	

unknown_23

unknown_24	

unknown_25

unknown_26	

unknown_27

unknown_28	

unknown_29

unknown_30	
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallfeatures_application_area_1"features_application_area_1_weightfeatures_application_area_2"features_application_area_2_weightfeatures_cowork_abroadfeatures_cowork_corfeatures_cowork_instfeatures_cowork_unifeatures_cowork_etcfeatures_econ_socialfeatures_green_techfeatures_log_durationfeatures_log_rnd_fundfeatures_multi_yearfeatures_n_patent_appfeatures_n_patent_regfeatures_n_of_korean_patentfeatures_n_of_paperfeatures_n_of_scifeatures_national_strategy_2features_rnd_orgfeatures_rnd_stagefeatures_stp_code_11features_stp_code_1_weightfeatures_stp_code_21features_stp_code_2_weightfeatures_sixt_2features_yearunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30*G
Tin@
>2<																							*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_dense_features_layer_call_and_return_conditional_losses_129036p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:??????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:d `
'
_output_shapes
:?????????
5
_user_specified_namefeatures/Application_Area_1:kg
'
_output_shapes
:?????????
<
_user_specified_name$"features/Application_Area_1_Weight:d`
'
_output_shapes
:?????????
5
_user_specified_namefeatures/Application_Area_2:kg
'
_output_shapes
:?????????
<
_user_specified_name$"features/Application_Area_2_Weight:_[
'
_output_shapes
:?????????
0
_user_specified_namefeatures/Cowork_Abroad:\X
'
_output_shapes
:?????????
-
_user_specified_namefeatures/Cowork_Cor:]Y
'
_output_shapes
:?????????
.
_user_specified_namefeatures/Cowork_Inst:\X
'
_output_shapes
:?????????
-
_user_specified_namefeatures/Cowork_Uni:\X
'
_output_shapes
:?????????
-
_user_specified_namefeatures/Cowork_etc:]	Y
'
_output_shapes
:?????????
.
_user_specified_namefeatures/Econ_Social:\
X
'
_output_shapes
:?????????
-
_user_specified_namefeatures/Green_Tech:^Z
'
_output_shapes
:?????????
/
_user_specified_namefeatures/Log_Duration:^Z
'
_output_shapes
:?????????
/
_user_specified_namefeatures/Log_RnD_Fund:\X
'
_output_shapes
:?????????
-
_user_specified_namefeatures/Multi_Year:^Z
'
_output_shapes
:?????????
/
_user_specified_namefeatures/N_Patent_App:^Z
'
_output_shapes
:?????????
/
_user_specified_namefeatures/N_Patent_Reg:d`
'
_output_shapes
:?????????
5
_user_specified_namefeatures/N_of_Korean_Patent:\X
'
_output_shapes
:?????????
-
_user_specified_namefeatures/N_of_Paper:ZV
'
_output_shapes
:?????????
+
_user_specified_namefeatures/N_of_SCI:ea
'
_output_shapes
:?????????
6
_user_specified_namefeatures/National_Strategy_2:YU
'
_output_shapes
:?????????
*
_user_specified_namefeatures/RnD_Org:[W
'
_output_shapes
:?????????
,
_user_specified_namefeatures/RnD_Stage:]Y
'
_output_shapes
:?????????
.
_user_specified_namefeatures/STP_Code_11:c_
'
_output_shapes
:?????????
4
_user_specified_namefeatures/STP_Code_1_Weight:]Y
'
_output_shapes
:?????????
.
_user_specified_namefeatures/STP_Code_21:c_
'
_output_shapes
:?????????
4
_user_specified_namefeatures/STP_Code_2_Weight:XT
'
_output_shapes
:?????????
)
_user_specified_namefeatures/SixT_2:VR
'
_output_shapes
:?????????
'
_user_specified_namefeatures/Year:

_output_shapes
: :

_output_shapes
: :!

_output_shapes
: :#

_output_shapes
: :%

_output_shapes
: :'

_output_shapes
: :)

_output_shapes
: :+

_output_shapes
: :-

_output_shapes
: :/

_output_shapes
: :1

_output_shapes
: :3

_output_shapes
: :5

_output_shapes
: :7

_output_shapes
: :9

_output_shapes
: :;

_output_shapes
: 
?
?
__inference_<lambda>_1339782
.table_init443_lookuptableimportv2_table_handle*
&table_init443_lookuptableimportv2_keys,
(table_init443_lookuptableimportv2_values	
identity??!table_init443/LookupTableImportV2?
!table_init443/LookupTableImportV2LookupTableImportV2.table_init443_lookuptableimportv2_table_handle&table_init443_lookuptableimportv2_keys(table_init443_lookuptableimportv2_values*	
Tin0*

Tout0	*
_output_shapes
 J
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??L
IdentityIdentityConst:output:0^NoOp*
T0*
_output_shapes
: j
NoOpNoOp"^table_init443/LookupTableImportV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*!
_input_shapes
: ::2F
!table_init443/LookupTableImportV2!table_init443/LookupTableImportV2: 

_output_shapes
:: 

_output_shapes
:
?3
?

+__inference_sequential_layer_call_fn_129275
application_area_1
application_area_1_weight
application_area_2
application_area_2_weight
cowork_abroad

cowork_cor
cowork_inst

cowork_uni

cowork_etc
econ_social	

green_tech	
log_duration
log_rnd_fund

multi_year	
n_patent_app
n_patent_reg
n_of_korean_patent

n_of_paper
n_of_sci
national_strategy_2	
rnd_org	
	rnd_stage	
stp_code_11
stp_code_1_weight
stp_code_21
stp_code_2_weight

sixt_2	
year
unknown
	unknown_0	
	unknown_1
	unknown_2	
	unknown_3
	unknown_4	
	unknown_5
	unknown_6	
	unknown_7
	unknown_8	
	unknown_9

unknown_10	

unknown_11

unknown_12	

unknown_13

unknown_14	

unknown_15

unknown_16	

unknown_17

unknown_18	

unknown_19

unknown_20	

unknown_21

unknown_22	

unknown_23

unknown_24	

unknown_25

unknown_26	

unknown_27

unknown_28	

unknown_29

unknown_30	

unknown_31:
??

unknown_32:	?

unknown_33:
??

unknown_34:	?

unknown_35:
??

unknown_36:	?

unknown_37:	?

unknown_38:
identity??StatefulPartitionedCall?	
StatefulPartitionedCallStatefulPartitionedCallapplication_area_1application_area_1_weightapplication_area_2application_area_2_weightcowork_abroad
cowork_corcowork_inst
cowork_uni
cowork_etcecon_social
green_techlog_durationlog_rnd_fund
multi_yearn_patent_appn_patent_regn_of_korean_patent
n_of_papern_of_scinational_strategy_2rnd_org	rnd_stagestp_code_11stp_code_1_weightstp_code_21stp_code_2_weightsixt_2yearunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38*O
TinH
F2D																							*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????**
_read_only_resource_inputs

<=>?@ABC*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_sequential_layer_call_and_return_conditional_losses_129192o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:[ W
'
_output_shapes
:?????????
,
_user_specified_nameApplication_Area_1:b^
'
_output_shapes
:?????????
3
_user_specified_nameApplication_Area_1_Weight:[W
'
_output_shapes
:?????????
,
_user_specified_nameApplication_Area_2:b^
'
_output_shapes
:?????????
3
_user_specified_nameApplication_Area_2_Weight:VR
'
_output_shapes
:?????????
'
_user_specified_nameCowork_Abroad:SO
'
_output_shapes
:?????????
$
_user_specified_name
Cowork_Cor:TP
'
_output_shapes
:?????????
%
_user_specified_nameCowork_Inst:SO
'
_output_shapes
:?????????
$
_user_specified_name
Cowork_Uni:SO
'
_output_shapes
:?????????
$
_user_specified_name
Cowork_etc:T	P
'
_output_shapes
:?????????
%
_user_specified_nameEcon_Social:S
O
'
_output_shapes
:?????????
$
_user_specified_name
Green_Tech:UQ
'
_output_shapes
:?????????
&
_user_specified_nameLog_Duration:UQ
'
_output_shapes
:?????????
&
_user_specified_nameLog_RnD_Fund:SO
'
_output_shapes
:?????????
$
_user_specified_name
Multi_Year:UQ
'
_output_shapes
:?????????
&
_user_specified_nameN_Patent_App:UQ
'
_output_shapes
:?????????
&
_user_specified_nameN_Patent_Reg:[W
'
_output_shapes
:?????????
,
_user_specified_nameN_of_Korean_Patent:SO
'
_output_shapes
:?????????
$
_user_specified_name
N_of_Paper:QM
'
_output_shapes
:?????????
"
_user_specified_name
N_of_SCI:\X
'
_output_shapes
:?????????
-
_user_specified_nameNational_Strategy_2:PL
'
_output_shapes
:?????????
!
_user_specified_name	RnD_Org:RN
'
_output_shapes
:?????????
#
_user_specified_name	RnD_Stage:TP
'
_output_shapes
:?????????
%
_user_specified_nameSTP_Code_11:ZV
'
_output_shapes
:?????????
+
_user_specified_nameSTP_Code_1_Weight:TP
'
_output_shapes
:?????????
%
_user_specified_nameSTP_Code_21:ZV
'
_output_shapes
:?????????
+
_user_specified_nameSTP_Code_2_Weight:OK
'
_output_shapes
:?????????
 
_user_specified_nameSixT_2:MI
'
_output_shapes
:?????????

_user_specified_nameYear:

_output_shapes
: :

_output_shapes
: :!

_output_shapes
: :#

_output_shapes
: :%

_output_shapes
: :'

_output_shapes
: :)

_output_shapes
: :+

_output_shapes
: :-

_output_shapes
: :/

_output_shapes
: :1

_output_shapes
: :3

_output_shapes
: :5

_output_shapes
: :7

_output_shapes
: :9

_output_shapes
: :;

_output_shapes
: 
?	
d
E__inference_dropout_2_layer_call_and_return_conditional_losses_133630

inputs
identity?R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??e
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:??????????C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????p
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????j
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:??????????Z
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:??????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
__inference__initializer_1338792
.table_init783_lookuptableimportv2_table_handle*
&table_init783_lookuptableimportv2_keys	,
(table_init783_lookuptableimportv2_values	
identity??!table_init783/LookupTableImportV2?
!table_init783/LookupTableImportV2LookupTableImportV2.table_init783_lookuptableimportv2_table_handle&table_init783_lookuptableimportv2_keys(table_init783_lookuptableimportv2_values*	
Tin0	*

Tout0	*
_output_shapes
 G
ConstConst*
_output_shapes
: *
dtype0*
value	B :L
IdentityIdentityConst:output:0^NoOp*
T0*
_output_shapes
: j
NoOpNoOp"^table_init783/LookupTableImportV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*!
_input_shapes
: ::2F
!table_init783/LookupTableImportV2!table_init783/LookupTableImportV2: 

_output_shapes
:: 

_output_shapes
:
?
c
E__inference_dropout_2_layer_call_and_return_conditional_losses_129172

inputs

identity_1O
IdentityIdentityinputs*
T0*(
_output_shapes
:??????????\

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:??????????"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
__inference__initializer_1336812
.table_init341_lookuptableimportv2_table_handle*
&table_init341_lookuptableimportv2_keys,
(table_init341_lookuptableimportv2_values	
identity??!table_init341/LookupTableImportV2?
!table_init341/LookupTableImportV2LookupTableImportV2.table_init341_lookuptableimportv2_table_handle&table_init341_lookuptableimportv2_keys(table_init341_lookuptableimportv2_values*	
Tin0*

Tout0	*
_output_shapes
 G
ConstConst*
_output_shapes
: *
dtype0*
value	B :L
IdentityIdentityConst:output:0^NoOp*
T0*
_output_shapes
: j
NoOpNoOp"^table_init341/LookupTableImportV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*!
_input_shapes
: :":"2F
!table_init341/LookupTableImportV2!table_init341/LookupTableImportV2: 

_output_shapes
:": 

_output_shapes
:"
??
?
J__inference_dense_features_layer_call_and_return_conditional_losses_129036
features

features_1

features_2

features_3

features_4

features_5

features_6

features_7

features_8

features_9	
features_10	
features_11
features_12
features_13	
features_14
features_15
features_16
features_17
features_18
features_19	
features_20	
features_21	
features_22
features_23
features_24
features_25
features_26	
features_27K
Gapplication_area_1_indicator_none_lookup_lookuptablefindv2_table_handleL
Happlication_area_1_indicator_none_lookup_lookuptablefindv2_default_value	K
Gapplication_area_2_indicator_none_lookup_lookuptablefindv2_table_handleL
Happlication_area_2_indicator_none_lookup_lookuptablefindv2_default_value	F
Bcowork_abroad_indicator_none_lookup_lookuptablefindv2_table_handleG
Ccowork_abroad_indicator_none_lookup_lookuptablefindv2_default_value	C
?cowork_cor_indicator_none_lookup_lookuptablefindv2_table_handleD
@cowork_cor_indicator_none_lookup_lookuptablefindv2_default_value	D
@cowork_inst_indicator_none_lookup_lookuptablefindv2_table_handleE
Acowork_inst_indicator_none_lookup_lookuptablefindv2_default_value	C
?cowork_uni_indicator_none_lookup_lookuptablefindv2_table_handleD
@cowork_uni_indicator_none_lookup_lookuptablefindv2_default_value	C
?cowork_etc_indicator_none_lookup_lookuptablefindv2_table_handleD
@cowork_etc_indicator_none_lookup_lookuptablefindv2_default_value	D
@econ_social_indicator_none_lookup_lookuptablefindv2_table_handleE
Aecon_social_indicator_none_lookup_lookuptablefindv2_default_value	C
?green_tech_indicator_none_lookup_lookuptablefindv2_table_handleD
@green_tech_indicator_none_lookup_lookuptablefindv2_default_value	C
?multi_year_indicator_none_lookup_lookuptablefindv2_table_handleD
@multi_year_indicator_none_lookup_lookuptablefindv2_default_value	L
Hnational_strategy_2_indicator_none_lookup_lookuptablefindv2_table_handleM
Inational_strategy_2_indicator_none_lookup_lookuptablefindv2_default_value	@
<rnd_org_indicator_none_lookup_lookuptablefindv2_table_handleA
=rnd_org_indicator_none_lookup_lookuptablefindv2_default_value	B
>rnd_stage_indicator_none_lookup_lookuptablefindv2_table_handleC
?rnd_stage_indicator_none_lookup_lookuptablefindv2_default_value	D
@stp_code_11_indicator_none_lookup_lookuptablefindv2_table_handleE
Astp_code_11_indicator_none_lookup_lookuptablefindv2_default_value	D
@stp_code_21_indicator_none_lookup_lookuptablefindv2_table_handleE
Astp_code_21_indicator_none_lookup_lookuptablefindv2_default_value	?
;sixt_2_indicator_none_lookup_lookuptablefindv2_table_handle@
<sixt_2_indicator_none_lookup_lookuptablefindv2_default_value	
identity??:Application_Area_1_indicator/None_Lookup/LookupTableFindV2?:Application_Area_2_indicator/None_Lookup/LookupTableFindV2?5Cowork_Abroad_indicator/None_Lookup/LookupTableFindV2?2Cowork_Cor_indicator/None_Lookup/LookupTableFindV2?3Cowork_Inst_indicator/None_Lookup/LookupTableFindV2?2Cowork_Uni_indicator/None_Lookup/LookupTableFindV2?2Cowork_etc_indicator/None_Lookup/LookupTableFindV2?3Econ_Social_indicator/None_Lookup/LookupTableFindV2?2Green_Tech_indicator/None_Lookup/LookupTableFindV2?2Multi_Year_indicator/None_Lookup/LookupTableFindV2?;National_Strategy_2_indicator/None_Lookup/LookupTableFindV2?/RnD_Org_indicator/None_Lookup/LookupTableFindV2?1RnD_Stage_indicator/None_Lookup/LookupTableFindV2?3STP_Code_11_indicator/None_Lookup/LookupTableFindV2?3STP_Code_21_indicator/None_Lookup/LookupTableFindV2?.SixT_2_indicator/None_Lookup/LookupTableFindV2Y
Application_Area_1_Weight/ShapeShape
features_1*
T0*
_output_shapes
:w
-Application_Area_1_Weight/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: y
/Application_Area_1_Weight/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:y
/Application_Area_1_Weight/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
'Application_Area_1_Weight/strided_sliceStridedSlice(Application_Area_1_Weight/Shape:output:06Application_Area_1_Weight/strided_slice/stack:output:08Application_Area_1_Weight/strided_slice/stack_1:output:08Application_Area_1_Weight/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskk
)Application_Area_1_Weight/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
'Application_Area_1_Weight/Reshape/shapePack0Application_Area_1_Weight/strided_slice:output:02Application_Area_1_Weight/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
!Application_Area_1_Weight/ReshapeReshape
features_10Application_Area_1_Weight/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????|
;Application_Area_1_indicator/to_sparse_input/ignore_value/xConst*
_output_shapes
: *
dtype0*
valueB B ?
5Application_Area_1_indicator/to_sparse_input/NotEqualNotEqualfeaturesDApplication_Area_1_indicator/to_sparse_input/ignore_value/x:output:0*
T0*'
_output_shapes
:??????????
4Application_Area_1_indicator/to_sparse_input/indicesWhere9Application_Area_1_indicator/to_sparse_input/NotEqual:z:0*'
_output_shapes
:??????????
3Application_Area_1_indicator/to_sparse_input/valuesGatherNdfeatures<Application_Area_1_indicator/to_sparse_input/indices:index:0*
Tindices0	*
Tparams0*#
_output_shapes
:??????????
8Application_Area_1_indicator/to_sparse_input/dense_shapeShapefeatures*
T0*
_output_shapes
:*
out_type0	?
:Application_Area_1_indicator/None_Lookup/LookupTableFindV2LookupTableFindV2Gapplication_area_1_indicator_none_lookup_lookuptablefindv2_table_handle<Application_Area_1_indicator/to_sparse_input/values:output:0Happlication_area_1_indicator_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:??????????
8Application_Area_1_indicator/SparseToDense/default_valueConst*
_output_shapes
: *
dtype0	*
valueB	 R
??????????
*Application_Area_1_indicator/SparseToDenseSparseToDense<Application_Area_1_indicator/to_sparse_input/indices:index:0AApplication_Area_1_indicator/to_sparse_input/dense_shape:output:0CApplication_Area_1_indicator/None_Lookup/LookupTableFindV2:values:0AApplication_Area_1_indicator/SparseToDense/default_value:output:0*
T0	*
Tindices0	*'
_output_shapes
:?????????o
*Application_Area_1_indicator/one_hot/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??q
,Application_Area_1_indicator/one_hot/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    l
*Application_Area_1_indicator/one_hot/depthConst*
_output_shapes
: *
dtype0*
value	B :!?
$Application_Area_1_indicator/one_hotOneHot2Application_Area_1_indicator/SparseToDense:dense:03Application_Area_1_indicator/one_hot/depth:output:03Application_Area_1_indicator/one_hot/Const:output:05Application_Area_1_indicator/one_hot/Const_1:output:0*
T0*+
_output_shapes
:?????????!?
2Application_Area_1_indicator/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
??????????
 Application_Area_1_indicator/SumSum-Application_Area_1_indicator/one_hot:output:0;Application_Area_1_indicator/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:?????????!{
"Application_Area_1_indicator/ShapeShape)Application_Area_1_indicator/Sum:output:0*
T0*
_output_shapes
:z
0Application_Area_1_indicator/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: |
2Application_Area_1_indicator/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:|
2Application_Area_1_indicator/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
*Application_Area_1_indicator/strided_sliceStridedSlice+Application_Area_1_indicator/Shape:output:09Application_Area_1_indicator/strided_slice/stack:output:0;Application_Area_1_indicator/strided_slice/stack_1:output:0;Application_Area_1_indicator/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskn
,Application_Area_1_indicator/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :!?
*Application_Area_1_indicator/Reshape/shapePack3Application_Area_1_indicator/strided_slice:output:05Application_Area_1_indicator/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
$Application_Area_1_indicator/ReshapeReshape)Application_Area_1_indicator/Sum:output:03Application_Area_1_indicator/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????!Y
Application_Area_2_Weight/ShapeShape
features_3*
T0*
_output_shapes
:w
-Application_Area_2_Weight/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: y
/Application_Area_2_Weight/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:y
/Application_Area_2_Weight/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
'Application_Area_2_Weight/strided_sliceStridedSlice(Application_Area_2_Weight/Shape:output:06Application_Area_2_Weight/strided_slice/stack:output:08Application_Area_2_Weight/strided_slice/stack_1:output:08Application_Area_2_Weight/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskk
)Application_Area_2_Weight/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
'Application_Area_2_Weight/Reshape/shapePack0Application_Area_2_Weight/strided_slice:output:02Application_Area_2_Weight/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
!Application_Area_2_Weight/ReshapeReshape
features_30Application_Area_2_Weight/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????|
;Application_Area_2_indicator/to_sparse_input/ignore_value/xConst*
_output_shapes
: *
dtype0*
valueB B ?
5Application_Area_2_indicator/to_sparse_input/NotEqualNotEqual
features_2DApplication_Area_2_indicator/to_sparse_input/ignore_value/x:output:0*
T0*'
_output_shapes
:??????????
4Application_Area_2_indicator/to_sparse_input/indicesWhere9Application_Area_2_indicator/to_sparse_input/NotEqual:z:0*'
_output_shapes
:??????????
3Application_Area_2_indicator/to_sparse_input/valuesGatherNd
features_2<Application_Area_2_indicator/to_sparse_input/indices:index:0*
Tindices0	*
Tparams0*#
_output_shapes
:??????????
8Application_Area_2_indicator/to_sparse_input/dense_shapeShape
features_2*
T0*
_output_shapes
:*
out_type0	?
:Application_Area_2_indicator/None_Lookup/LookupTableFindV2LookupTableFindV2Gapplication_area_2_indicator_none_lookup_lookuptablefindv2_table_handle<Application_Area_2_indicator/to_sparse_input/values:output:0Happlication_area_2_indicator_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:??????????
8Application_Area_2_indicator/SparseToDense/default_valueConst*
_output_shapes
: *
dtype0	*
valueB	 R
??????????
*Application_Area_2_indicator/SparseToDenseSparseToDense<Application_Area_2_indicator/to_sparse_input/indices:index:0AApplication_Area_2_indicator/to_sparse_input/dense_shape:output:0CApplication_Area_2_indicator/None_Lookup/LookupTableFindV2:values:0AApplication_Area_2_indicator/SparseToDense/default_value:output:0*
T0	*
Tindices0	*'
_output_shapes
:?????????o
*Application_Area_2_indicator/one_hot/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??q
,Application_Area_2_indicator/one_hot/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    l
*Application_Area_2_indicator/one_hot/depthConst*
_output_shapes
: *
dtype0*
value	B :"?
$Application_Area_2_indicator/one_hotOneHot2Application_Area_2_indicator/SparseToDense:dense:03Application_Area_2_indicator/one_hot/depth:output:03Application_Area_2_indicator/one_hot/Const:output:05Application_Area_2_indicator/one_hot/Const_1:output:0*
T0*+
_output_shapes
:?????????"?
2Application_Area_2_indicator/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
??????????
 Application_Area_2_indicator/SumSum-Application_Area_2_indicator/one_hot:output:0;Application_Area_2_indicator/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:?????????"{
"Application_Area_2_indicator/ShapeShape)Application_Area_2_indicator/Sum:output:0*
T0*
_output_shapes
:z
0Application_Area_2_indicator/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: |
2Application_Area_2_indicator/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:|
2Application_Area_2_indicator/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
*Application_Area_2_indicator/strided_sliceStridedSlice+Application_Area_2_indicator/Shape:output:09Application_Area_2_indicator/strided_slice/stack:output:0;Application_Area_2_indicator/strided_slice/stack_1:output:0;Application_Area_2_indicator/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskn
,Application_Area_2_indicator/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :"?
*Application_Area_2_indicator/Reshape/shapePack3Application_Area_2_indicator/strided_slice:output:05Application_Area_2_indicator/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
$Application_Area_2_indicator/ReshapeReshape)Application_Area_2_indicator/Sum:output:03Application_Area_2_indicator/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????"w
6Cowork_Abroad_indicator/to_sparse_input/ignore_value/xConst*
_output_shapes
: *
dtype0*
valueB B ?
0Cowork_Abroad_indicator/to_sparse_input/NotEqualNotEqual
features_4?Cowork_Abroad_indicator/to_sparse_input/ignore_value/x:output:0*
T0*'
_output_shapes
:??????????
/Cowork_Abroad_indicator/to_sparse_input/indicesWhere4Cowork_Abroad_indicator/to_sparse_input/NotEqual:z:0*'
_output_shapes
:??????????
.Cowork_Abroad_indicator/to_sparse_input/valuesGatherNd
features_47Cowork_Abroad_indicator/to_sparse_input/indices:index:0*
Tindices0	*
Tparams0*#
_output_shapes
:?????????}
3Cowork_Abroad_indicator/to_sparse_input/dense_shapeShape
features_4*
T0*
_output_shapes
:*
out_type0	?
5Cowork_Abroad_indicator/None_Lookup/LookupTableFindV2LookupTableFindV2Bcowork_abroad_indicator_none_lookup_lookuptablefindv2_table_handle7Cowork_Abroad_indicator/to_sparse_input/values:output:0Ccowork_abroad_indicator_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:?????????~
3Cowork_Abroad_indicator/SparseToDense/default_valueConst*
_output_shapes
: *
dtype0	*
valueB	 R
??????????
%Cowork_Abroad_indicator/SparseToDenseSparseToDense7Cowork_Abroad_indicator/to_sparse_input/indices:index:0<Cowork_Abroad_indicator/to_sparse_input/dense_shape:output:0>Cowork_Abroad_indicator/None_Lookup/LookupTableFindV2:values:0<Cowork_Abroad_indicator/SparseToDense/default_value:output:0*
T0	*
Tindices0	*'
_output_shapes
:?????????j
%Cowork_Abroad_indicator/one_hot/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??l
'Cowork_Abroad_indicator/one_hot/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    g
%Cowork_Abroad_indicator/one_hot/depthConst*
_output_shapes
: *
dtype0*
value	B :?
Cowork_Abroad_indicator/one_hotOneHot-Cowork_Abroad_indicator/SparseToDense:dense:0.Cowork_Abroad_indicator/one_hot/depth:output:0.Cowork_Abroad_indicator/one_hot/Const:output:00Cowork_Abroad_indicator/one_hot/Const_1:output:0*
T0*+
_output_shapes
:??????????
-Cowork_Abroad_indicator/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
??????????
Cowork_Abroad_indicator/SumSum(Cowork_Abroad_indicator/one_hot:output:06Cowork_Abroad_indicator/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:?????????q
Cowork_Abroad_indicator/ShapeShape$Cowork_Abroad_indicator/Sum:output:0*
T0*
_output_shapes
:u
+Cowork_Abroad_indicator/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: w
-Cowork_Abroad_indicator/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:w
-Cowork_Abroad_indicator/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
%Cowork_Abroad_indicator/strided_sliceStridedSlice&Cowork_Abroad_indicator/Shape:output:04Cowork_Abroad_indicator/strided_slice/stack:output:06Cowork_Abroad_indicator/strided_slice/stack_1:output:06Cowork_Abroad_indicator/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maski
'Cowork_Abroad_indicator/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
%Cowork_Abroad_indicator/Reshape/shapePack.Cowork_Abroad_indicator/strided_slice:output:00Cowork_Abroad_indicator/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
Cowork_Abroad_indicator/ReshapeReshape$Cowork_Abroad_indicator/Sum:output:0.Cowork_Abroad_indicator/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????t
3Cowork_Cor_indicator/to_sparse_input/ignore_value/xConst*
_output_shapes
: *
dtype0*
valueB B ?
-Cowork_Cor_indicator/to_sparse_input/NotEqualNotEqual
features_5<Cowork_Cor_indicator/to_sparse_input/ignore_value/x:output:0*
T0*'
_output_shapes
:??????????
,Cowork_Cor_indicator/to_sparse_input/indicesWhere1Cowork_Cor_indicator/to_sparse_input/NotEqual:z:0*'
_output_shapes
:??????????
+Cowork_Cor_indicator/to_sparse_input/valuesGatherNd
features_54Cowork_Cor_indicator/to_sparse_input/indices:index:0*
Tindices0	*
Tparams0*#
_output_shapes
:?????????z
0Cowork_Cor_indicator/to_sparse_input/dense_shapeShape
features_5*
T0*
_output_shapes
:*
out_type0	?
2Cowork_Cor_indicator/None_Lookup/LookupTableFindV2LookupTableFindV2?cowork_cor_indicator_none_lookup_lookuptablefindv2_table_handle4Cowork_Cor_indicator/to_sparse_input/values:output:0@cowork_cor_indicator_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:?????????{
0Cowork_Cor_indicator/SparseToDense/default_valueConst*
_output_shapes
: *
dtype0	*
valueB	 R
??????????
"Cowork_Cor_indicator/SparseToDenseSparseToDense4Cowork_Cor_indicator/to_sparse_input/indices:index:09Cowork_Cor_indicator/to_sparse_input/dense_shape:output:0;Cowork_Cor_indicator/None_Lookup/LookupTableFindV2:values:09Cowork_Cor_indicator/SparseToDense/default_value:output:0*
T0	*
Tindices0	*'
_output_shapes
:?????????g
"Cowork_Cor_indicator/one_hot/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??i
$Cowork_Cor_indicator/one_hot/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    d
"Cowork_Cor_indicator/one_hot/depthConst*
_output_shapes
: *
dtype0*
value	B :?
Cowork_Cor_indicator/one_hotOneHot*Cowork_Cor_indicator/SparseToDense:dense:0+Cowork_Cor_indicator/one_hot/depth:output:0+Cowork_Cor_indicator/one_hot/Const:output:0-Cowork_Cor_indicator/one_hot/Const_1:output:0*
T0*+
_output_shapes
:?????????}
*Cowork_Cor_indicator/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
??????????
Cowork_Cor_indicator/SumSum%Cowork_Cor_indicator/one_hot:output:03Cowork_Cor_indicator/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:?????????k
Cowork_Cor_indicator/ShapeShape!Cowork_Cor_indicator/Sum:output:0*
T0*
_output_shapes
:r
(Cowork_Cor_indicator/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: t
*Cowork_Cor_indicator/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:t
*Cowork_Cor_indicator/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
"Cowork_Cor_indicator/strided_sliceStridedSlice#Cowork_Cor_indicator/Shape:output:01Cowork_Cor_indicator/strided_slice/stack:output:03Cowork_Cor_indicator/strided_slice/stack_1:output:03Cowork_Cor_indicator/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
$Cowork_Cor_indicator/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
"Cowork_Cor_indicator/Reshape/shapePack+Cowork_Cor_indicator/strided_slice:output:0-Cowork_Cor_indicator/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
Cowork_Cor_indicator/ReshapeReshape!Cowork_Cor_indicator/Sum:output:0+Cowork_Cor_indicator/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????u
4Cowork_Inst_indicator/to_sparse_input/ignore_value/xConst*
_output_shapes
: *
dtype0*
valueB B ?
.Cowork_Inst_indicator/to_sparse_input/NotEqualNotEqual
features_6=Cowork_Inst_indicator/to_sparse_input/ignore_value/x:output:0*
T0*'
_output_shapes
:??????????
-Cowork_Inst_indicator/to_sparse_input/indicesWhere2Cowork_Inst_indicator/to_sparse_input/NotEqual:z:0*'
_output_shapes
:??????????
,Cowork_Inst_indicator/to_sparse_input/valuesGatherNd
features_65Cowork_Inst_indicator/to_sparse_input/indices:index:0*
Tindices0	*
Tparams0*#
_output_shapes
:?????????{
1Cowork_Inst_indicator/to_sparse_input/dense_shapeShape
features_6*
T0*
_output_shapes
:*
out_type0	?
3Cowork_Inst_indicator/None_Lookup/LookupTableFindV2LookupTableFindV2@cowork_inst_indicator_none_lookup_lookuptablefindv2_table_handle5Cowork_Inst_indicator/to_sparse_input/values:output:0Acowork_inst_indicator_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:?????????|
1Cowork_Inst_indicator/SparseToDense/default_valueConst*
_output_shapes
: *
dtype0	*
valueB	 R
??????????
#Cowork_Inst_indicator/SparseToDenseSparseToDense5Cowork_Inst_indicator/to_sparse_input/indices:index:0:Cowork_Inst_indicator/to_sparse_input/dense_shape:output:0<Cowork_Inst_indicator/None_Lookup/LookupTableFindV2:values:0:Cowork_Inst_indicator/SparseToDense/default_value:output:0*
T0	*
Tindices0	*'
_output_shapes
:?????????h
#Cowork_Inst_indicator/one_hot/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??j
%Cowork_Inst_indicator/one_hot/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    e
#Cowork_Inst_indicator/one_hot/depthConst*
_output_shapes
: *
dtype0*
value	B :?
Cowork_Inst_indicator/one_hotOneHot+Cowork_Inst_indicator/SparseToDense:dense:0,Cowork_Inst_indicator/one_hot/depth:output:0,Cowork_Inst_indicator/one_hot/Const:output:0.Cowork_Inst_indicator/one_hot/Const_1:output:0*
T0*+
_output_shapes
:?????????~
+Cowork_Inst_indicator/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
??????????
Cowork_Inst_indicator/SumSum&Cowork_Inst_indicator/one_hot:output:04Cowork_Inst_indicator/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:?????????m
Cowork_Inst_indicator/ShapeShape"Cowork_Inst_indicator/Sum:output:0*
T0*
_output_shapes
:s
)Cowork_Inst_indicator/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: u
+Cowork_Inst_indicator/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+Cowork_Inst_indicator/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
#Cowork_Inst_indicator/strided_sliceStridedSlice$Cowork_Inst_indicator/Shape:output:02Cowork_Inst_indicator/strided_slice/stack:output:04Cowork_Inst_indicator/strided_slice/stack_1:output:04Cowork_Inst_indicator/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskg
%Cowork_Inst_indicator/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
#Cowork_Inst_indicator/Reshape/shapePack,Cowork_Inst_indicator/strided_slice:output:0.Cowork_Inst_indicator/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
Cowork_Inst_indicator/ReshapeReshape"Cowork_Inst_indicator/Sum:output:0,Cowork_Inst_indicator/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????t
3Cowork_Uni_indicator/to_sparse_input/ignore_value/xConst*
_output_shapes
: *
dtype0*
valueB B ?
-Cowork_Uni_indicator/to_sparse_input/NotEqualNotEqual
features_7<Cowork_Uni_indicator/to_sparse_input/ignore_value/x:output:0*
T0*'
_output_shapes
:??????????
,Cowork_Uni_indicator/to_sparse_input/indicesWhere1Cowork_Uni_indicator/to_sparse_input/NotEqual:z:0*'
_output_shapes
:??????????
+Cowork_Uni_indicator/to_sparse_input/valuesGatherNd
features_74Cowork_Uni_indicator/to_sparse_input/indices:index:0*
Tindices0	*
Tparams0*#
_output_shapes
:?????????z
0Cowork_Uni_indicator/to_sparse_input/dense_shapeShape
features_7*
T0*
_output_shapes
:*
out_type0	?
2Cowork_Uni_indicator/None_Lookup/LookupTableFindV2LookupTableFindV2?cowork_uni_indicator_none_lookup_lookuptablefindv2_table_handle4Cowork_Uni_indicator/to_sparse_input/values:output:0@cowork_uni_indicator_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:?????????{
0Cowork_Uni_indicator/SparseToDense/default_valueConst*
_output_shapes
: *
dtype0	*
valueB	 R
??????????
"Cowork_Uni_indicator/SparseToDenseSparseToDense4Cowork_Uni_indicator/to_sparse_input/indices:index:09Cowork_Uni_indicator/to_sparse_input/dense_shape:output:0;Cowork_Uni_indicator/None_Lookup/LookupTableFindV2:values:09Cowork_Uni_indicator/SparseToDense/default_value:output:0*
T0	*
Tindices0	*'
_output_shapes
:?????????g
"Cowork_Uni_indicator/one_hot/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??i
$Cowork_Uni_indicator/one_hot/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    d
"Cowork_Uni_indicator/one_hot/depthConst*
_output_shapes
: *
dtype0*
value	B :?
Cowork_Uni_indicator/one_hotOneHot*Cowork_Uni_indicator/SparseToDense:dense:0+Cowork_Uni_indicator/one_hot/depth:output:0+Cowork_Uni_indicator/one_hot/Const:output:0-Cowork_Uni_indicator/one_hot/Const_1:output:0*
T0*+
_output_shapes
:?????????}
*Cowork_Uni_indicator/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
??????????
Cowork_Uni_indicator/SumSum%Cowork_Uni_indicator/one_hot:output:03Cowork_Uni_indicator/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:?????????k
Cowork_Uni_indicator/ShapeShape!Cowork_Uni_indicator/Sum:output:0*
T0*
_output_shapes
:r
(Cowork_Uni_indicator/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: t
*Cowork_Uni_indicator/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:t
*Cowork_Uni_indicator/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
"Cowork_Uni_indicator/strided_sliceStridedSlice#Cowork_Uni_indicator/Shape:output:01Cowork_Uni_indicator/strided_slice/stack:output:03Cowork_Uni_indicator/strided_slice/stack_1:output:03Cowork_Uni_indicator/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
$Cowork_Uni_indicator/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
"Cowork_Uni_indicator/Reshape/shapePack+Cowork_Uni_indicator/strided_slice:output:0-Cowork_Uni_indicator/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
Cowork_Uni_indicator/ReshapeReshape!Cowork_Uni_indicator/Sum:output:0+Cowork_Uni_indicator/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????t
3Cowork_etc_indicator/to_sparse_input/ignore_value/xConst*
_output_shapes
: *
dtype0*
valueB B ?
-Cowork_etc_indicator/to_sparse_input/NotEqualNotEqual
features_8<Cowork_etc_indicator/to_sparse_input/ignore_value/x:output:0*
T0*'
_output_shapes
:??????????
,Cowork_etc_indicator/to_sparse_input/indicesWhere1Cowork_etc_indicator/to_sparse_input/NotEqual:z:0*'
_output_shapes
:??????????
+Cowork_etc_indicator/to_sparse_input/valuesGatherNd
features_84Cowork_etc_indicator/to_sparse_input/indices:index:0*
Tindices0	*
Tparams0*#
_output_shapes
:?????????z
0Cowork_etc_indicator/to_sparse_input/dense_shapeShape
features_8*
T0*
_output_shapes
:*
out_type0	?
2Cowork_etc_indicator/None_Lookup/LookupTableFindV2LookupTableFindV2?cowork_etc_indicator_none_lookup_lookuptablefindv2_table_handle4Cowork_etc_indicator/to_sparse_input/values:output:0@cowork_etc_indicator_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:?????????{
0Cowork_etc_indicator/SparseToDense/default_valueConst*
_output_shapes
: *
dtype0	*
valueB	 R
??????????
"Cowork_etc_indicator/SparseToDenseSparseToDense4Cowork_etc_indicator/to_sparse_input/indices:index:09Cowork_etc_indicator/to_sparse_input/dense_shape:output:0;Cowork_etc_indicator/None_Lookup/LookupTableFindV2:values:09Cowork_etc_indicator/SparseToDense/default_value:output:0*
T0	*
Tindices0	*'
_output_shapes
:?????????g
"Cowork_etc_indicator/one_hot/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??i
$Cowork_etc_indicator/one_hot/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    d
"Cowork_etc_indicator/one_hot/depthConst*
_output_shapes
: *
dtype0*
value	B :?
Cowork_etc_indicator/one_hotOneHot*Cowork_etc_indicator/SparseToDense:dense:0+Cowork_etc_indicator/one_hot/depth:output:0+Cowork_etc_indicator/one_hot/Const:output:0-Cowork_etc_indicator/one_hot/Const_1:output:0*
T0*+
_output_shapes
:?????????}
*Cowork_etc_indicator/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
??????????
Cowork_etc_indicator/SumSum%Cowork_etc_indicator/one_hot:output:03Cowork_etc_indicator/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:?????????k
Cowork_etc_indicator/ShapeShape!Cowork_etc_indicator/Sum:output:0*
T0*
_output_shapes
:r
(Cowork_etc_indicator/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: t
*Cowork_etc_indicator/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:t
*Cowork_etc_indicator/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
"Cowork_etc_indicator/strided_sliceStridedSlice#Cowork_etc_indicator/Shape:output:01Cowork_etc_indicator/strided_slice/stack:output:03Cowork_etc_indicator/strided_slice/stack_1:output:03Cowork_etc_indicator/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
$Cowork_etc_indicator/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
"Cowork_etc_indicator/Reshape/shapePack+Cowork_etc_indicator/strided_slice:output:0-Cowork_etc_indicator/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
Cowork_etc_indicator/ReshapeReshape!Cowork_etc_indicator/Sum:output:0+Cowork_etc_indicator/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????
4Econ_Social_indicator/to_sparse_input/ignore_value/xConst*
_output_shapes
: *
dtype0*
valueB :
??????????
2Econ_Social_indicator/to_sparse_input/ignore_valueCast=Econ_Social_indicator/to_sparse_input/ignore_value/x:output:0*

DstT0	*

SrcT0*
_output_shapes
: ?
.Econ_Social_indicator/to_sparse_input/NotEqualNotEqual
features_96Econ_Social_indicator/to_sparse_input/ignore_value:y:0*
T0	*'
_output_shapes
:??????????
-Econ_Social_indicator/to_sparse_input/indicesWhere2Econ_Social_indicator/to_sparse_input/NotEqual:z:0*'
_output_shapes
:??????????
,Econ_Social_indicator/to_sparse_input/valuesGatherNd
features_95Econ_Social_indicator/to_sparse_input/indices:index:0*
Tindices0	*
Tparams0	*#
_output_shapes
:?????????{
1Econ_Social_indicator/to_sparse_input/dense_shapeShape
features_9*
T0	*
_output_shapes
:*
out_type0	?
3Econ_Social_indicator/None_Lookup/LookupTableFindV2LookupTableFindV2@econ_social_indicator_none_lookup_lookuptablefindv2_table_handle5Econ_Social_indicator/to_sparse_input/values:output:0Aecon_social_indicator_none_lookup_lookuptablefindv2_default_value*	
Tin0	*

Tout0	*#
_output_shapes
:?????????|
1Econ_Social_indicator/SparseToDense/default_valueConst*
_output_shapes
: *
dtype0	*
valueB	 R
??????????
#Econ_Social_indicator/SparseToDenseSparseToDense5Econ_Social_indicator/to_sparse_input/indices:index:0:Econ_Social_indicator/to_sparse_input/dense_shape:output:0<Econ_Social_indicator/None_Lookup/LookupTableFindV2:values:0:Econ_Social_indicator/SparseToDense/default_value:output:0*
T0	*
Tindices0	*'
_output_shapes
:?????????h
#Econ_Social_indicator/one_hot/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??j
%Econ_Social_indicator/one_hot/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    e
#Econ_Social_indicator/one_hot/depthConst*
_output_shapes
: *
dtype0*
value	B :?
Econ_Social_indicator/one_hotOneHot+Econ_Social_indicator/SparseToDense:dense:0,Econ_Social_indicator/one_hot/depth:output:0,Econ_Social_indicator/one_hot/Const:output:0.Econ_Social_indicator/one_hot/Const_1:output:0*
T0*+
_output_shapes
:?????????~
+Econ_Social_indicator/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
??????????
Econ_Social_indicator/SumSum&Econ_Social_indicator/one_hot:output:04Econ_Social_indicator/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:?????????m
Econ_Social_indicator/ShapeShape"Econ_Social_indicator/Sum:output:0*
T0*
_output_shapes
:s
)Econ_Social_indicator/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: u
+Econ_Social_indicator/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+Econ_Social_indicator/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
#Econ_Social_indicator/strided_sliceStridedSlice$Econ_Social_indicator/Shape:output:02Econ_Social_indicator/strided_slice/stack:output:04Econ_Social_indicator/strided_slice/stack_1:output:04Econ_Social_indicator/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskg
%Econ_Social_indicator/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
#Econ_Social_indicator/Reshape/shapePack,Econ_Social_indicator/strided_slice:output:0.Econ_Social_indicator/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
Econ_Social_indicator/ReshapeReshape"Econ_Social_indicator/Sum:output:0,Econ_Social_indicator/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????~
3Green_Tech_indicator/to_sparse_input/ignore_value/xConst*
_output_shapes
: *
dtype0*
valueB :
??????????
1Green_Tech_indicator/to_sparse_input/ignore_valueCast<Green_Tech_indicator/to_sparse_input/ignore_value/x:output:0*

DstT0	*

SrcT0*
_output_shapes
: ?
-Green_Tech_indicator/to_sparse_input/NotEqualNotEqualfeatures_105Green_Tech_indicator/to_sparse_input/ignore_value:y:0*
T0	*'
_output_shapes
:??????????
,Green_Tech_indicator/to_sparse_input/indicesWhere1Green_Tech_indicator/to_sparse_input/NotEqual:z:0*'
_output_shapes
:??????????
+Green_Tech_indicator/to_sparse_input/valuesGatherNdfeatures_104Green_Tech_indicator/to_sparse_input/indices:index:0*
Tindices0	*
Tparams0	*#
_output_shapes
:?????????{
0Green_Tech_indicator/to_sparse_input/dense_shapeShapefeatures_10*
T0	*
_output_shapes
:*
out_type0	?
2Green_Tech_indicator/None_Lookup/LookupTableFindV2LookupTableFindV2?green_tech_indicator_none_lookup_lookuptablefindv2_table_handle4Green_Tech_indicator/to_sparse_input/values:output:0@green_tech_indicator_none_lookup_lookuptablefindv2_default_value*	
Tin0	*

Tout0	*#
_output_shapes
:?????????{
0Green_Tech_indicator/SparseToDense/default_valueConst*
_output_shapes
: *
dtype0	*
valueB	 R
??????????
"Green_Tech_indicator/SparseToDenseSparseToDense4Green_Tech_indicator/to_sparse_input/indices:index:09Green_Tech_indicator/to_sparse_input/dense_shape:output:0;Green_Tech_indicator/None_Lookup/LookupTableFindV2:values:09Green_Tech_indicator/SparseToDense/default_value:output:0*
T0	*
Tindices0	*'
_output_shapes
:?????????g
"Green_Tech_indicator/one_hot/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??i
$Green_Tech_indicator/one_hot/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    d
"Green_Tech_indicator/one_hot/depthConst*
_output_shapes
: *
dtype0*
value	B :<?
Green_Tech_indicator/one_hotOneHot*Green_Tech_indicator/SparseToDense:dense:0+Green_Tech_indicator/one_hot/depth:output:0+Green_Tech_indicator/one_hot/Const:output:0-Green_Tech_indicator/one_hot/Const_1:output:0*
T0*+
_output_shapes
:?????????<}
*Green_Tech_indicator/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
??????????
Green_Tech_indicator/SumSum%Green_Tech_indicator/one_hot:output:03Green_Tech_indicator/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:?????????<k
Green_Tech_indicator/ShapeShape!Green_Tech_indicator/Sum:output:0*
T0*
_output_shapes
:r
(Green_Tech_indicator/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: t
*Green_Tech_indicator/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:t
*Green_Tech_indicator/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
"Green_Tech_indicator/strided_sliceStridedSlice#Green_Tech_indicator/Shape:output:01Green_Tech_indicator/strided_slice/stack:output:03Green_Tech_indicator/strided_slice/stack_1:output:03Green_Tech_indicator/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
$Green_Tech_indicator/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :<?
"Green_Tech_indicator/Reshape/shapePack+Green_Tech_indicator/strided_slice:output:0-Green_Tech_indicator/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
Green_Tech_indicator/ReshapeReshape!Green_Tech_indicator/Sum:output:0+Green_Tech_indicator/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????<M
Log_Duration/ShapeShapefeatures_11*
T0*
_output_shapes
:j
 Log_Duration/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: l
"Log_Duration/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:l
"Log_Duration/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
Log_Duration/strided_sliceStridedSliceLog_Duration/Shape:output:0)Log_Duration/strided_slice/stack:output:0+Log_Duration/strided_slice/stack_1:output:0+Log_Duration/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask^
Log_Duration/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
Log_Duration/Reshape/shapePack#Log_Duration/strided_slice:output:0%Log_Duration/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
Log_Duration/ReshapeReshapefeatures_11#Log_Duration/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????M
Log_RnD_Fund/ShapeShapefeatures_12*
T0*
_output_shapes
:j
 Log_RnD_Fund/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: l
"Log_RnD_Fund/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:l
"Log_RnD_Fund/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
Log_RnD_Fund/strided_sliceStridedSliceLog_RnD_Fund/Shape:output:0)Log_RnD_Fund/strided_slice/stack:output:0+Log_RnD_Fund/strided_slice/stack_1:output:0+Log_RnD_Fund/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask^
Log_RnD_Fund/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
Log_RnD_Fund/Reshape/shapePack#Log_RnD_Fund/strided_slice:output:0%Log_RnD_Fund/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
Log_RnD_Fund/ReshapeReshapefeatures_12#Log_RnD_Fund/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????~
3Multi_Year_indicator/to_sparse_input/ignore_value/xConst*
_output_shapes
: *
dtype0*
valueB :
??????????
1Multi_Year_indicator/to_sparse_input/ignore_valueCast<Multi_Year_indicator/to_sparse_input/ignore_value/x:output:0*

DstT0	*

SrcT0*
_output_shapes
: ?
-Multi_Year_indicator/to_sparse_input/NotEqualNotEqualfeatures_135Multi_Year_indicator/to_sparse_input/ignore_value:y:0*
T0	*'
_output_shapes
:??????????
,Multi_Year_indicator/to_sparse_input/indicesWhere1Multi_Year_indicator/to_sparse_input/NotEqual:z:0*'
_output_shapes
:??????????
+Multi_Year_indicator/to_sparse_input/valuesGatherNdfeatures_134Multi_Year_indicator/to_sparse_input/indices:index:0*
Tindices0	*
Tparams0	*#
_output_shapes
:?????????{
0Multi_Year_indicator/to_sparse_input/dense_shapeShapefeatures_13*
T0	*
_output_shapes
:*
out_type0	?
2Multi_Year_indicator/None_Lookup/LookupTableFindV2LookupTableFindV2?multi_year_indicator_none_lookup_lookuptablefindv2_table_handle4Multi_Year_indicator/to_sparse_input/values:output:0@multi_year_indicator_none_lookup_lookuptablefindv2_default_value*	
Tin0	*

Tout0	*#
_output_shapes
:?????????{
0Multi_Year_indicator/SparseToDense/default_valueConst*
_output_shapes
: *
dtype0	*
valueB	 R
??????????
"Multi_Year_indicator/SparseToDenseSparseToDense4Multi_Year_indicator/to_sparse_input/indices:index:09Multi_Year_indicator/to_sparse_input/dense_shape:output:0;Multi_Year_indicator/None_Lookup/LookupTableFindV2:values:09Multi_Year_indicator/SparseToDense/default_value:output:0*
T0	*
Tindices0	*'
_output_shapes
:?????????g
"Multi_Year_indicator/one_hot/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??i
$Multi_Year_indicator/one_hot/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    d
"Multi_Year_indicator/one_hot/depthConst*
_output_shapes
: *
dtype0*
value	B :?
Multi_Year_indicator/one_hotOneHot*Multi_Year_indicator/SparseToDense:dense:0+Multi_Year_indicator/one_hot/depth:output:0+Multi_Year_indicator/one_hot/Const:output:0-Multi_Year_indicator/one_hot/Const_1:output:0*
T0*+
_output_shapes
:?????????}
*Multi_Year_indicator/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
??????????
Multi_Year_indicator/SumSum%Multi_Year_indicator/one_hot:output:03Multi_Year_indicator/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:?????????k
Multi_Year_indicator/ShapeShape!Multi_Year_indicator/Sum:output:0*
T0*
_output_shapes
:r
(Multi_Year_indicator/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: t
*Multi_Year_indicator/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:t
*Multi_Year_indicator/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
"Multi_Year_indicator/strided_sliceStridedSlice#Multi_Year_indicator/Shape:output:01Multi_Year_indicator/strided_slice/stack:output:03Multi_Year_indicator/strided_slice/stack_1:output:03Multi_Year_indicator/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
$Multi_Year_indicator/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
"Multi_Year_indicator/Reshape/shapePack+Multi_Year_indicator/strided_slice:output:0-Multi_Year_indicator/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
Multi_Year_indicator/ReshapeReshape!Multi_Year_indicator/Sum:output:0+Multi_Year_indicator/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????M
N_Patent_App/ShapeShapefeatures_14*
T0*
_output_shapes
:j
 N_Patent_App/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: l
"N_Patent_App/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:l
"N_Patent_App/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
N_Patent_App/strided_sliceStridedSliceN_Patent_App/Shape:output:0)N_Patent_App/strided_slice/stack:output:0+N_Patent_App/strided_slice/stack_1:output:0+N_Patent_App/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask^
N_Patent_App/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
N_Patent_App/Reshape/shapePack#N_Patent_App/strided_slice:output:0%N_Patent_App/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
N_Patent_App/ReshapeReshapefeatures_14#N_Patent_App/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????M
N_Patent_Reg/ShapeShapefeatures_15*
T0*
_output_shapes
:j
 N_Patent_Reg/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: l
"N_Patent_Reg/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:l
"N_Patent_Reg/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
N_Patent_Reg/strided_sliceStridedSliceN_Patent_Reg/Shape:output:0)N_Patent_Reg/strided_slice/stack:output:0+N_Patent_Reg/strided_slice/stack_1:output:0+N_Patent_Reg/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask^
N_Patent_Reg/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
N_Patent_Reg/Reshape/shapePack#N_Patent_Reg/strided_slice:output:0%N_Patent_Reg/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
N_Patent_Reg/ReshapeReshapefeatures_15#N_Patent_Reg/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????S
N_of_Korean_Patent/ShapeShapefeatures_16*
T0*
_output_shapes
:p
&N_of_Korean_Patent/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: r
(N_of_Korean_Patent/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(N_of_Korean_Patent/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
 N_of_Korean_Patent/strided_sliceStridedSlice!N_of_Korean_Patent/Shape:output:0/N_of_Korean_Patent/strided_slice/stack:output:01N_of_Korean_Patent/strided_slice/stack_1:output:01N_of_Korean_Patent/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskd
"N_of_Korean_Patent/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
 N_of_Korean_Patent/Reshape/shapePack)N_of_Korean_Patent/strided_slice:output:0+N_of_Korean_Patent/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
N_of_Korean_Patent/ReshapeReshapefeatures_16)N_of_Korean_Patent/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????K
N_of_Paper/ShapeShapefeatures_17*
T0*
_output_shapes
:h
N_of_Paper/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: j
 N_of_Paper/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:j
 N_of_Paper/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
N_of_Paper/strided_sliceStridedSliceN_of_Paper/Shape:output:0'N_of_Paper/strided_slice/stack:output:0)N_of_Paper/strided_slice/stack_1:output:0)N_of_Paper/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask\
N_of_Paper/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
N_of_Paper/Reshape/shapePack!N_of_Paper/strided_slice:output:0#N_of_Paper/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:
N_of_Paper/ReshapeReshapefeatures_17!N_of_Paper/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????I
N_of_SCI/ShapeShapefeatures_18*
T0*
_output_shapes
:f
N_of_SCI/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: h
N_of_SCI/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:h
N_of_SCI/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
N_of_SCI/strided_sliceStridedSliceN_of_SCI/Shape:output:0%N_of_SCI/strided_slice/stack:output:0'N_of_SCI/strided_slice/stack_1:output:0'N_of_SCI/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskZ
N_of_SCI/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
N_of_SCI/Reshape/shapePackN_of_SCI/strided_slice:output:0!N_of_SCI/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:{
N_of_SCI/ReshapeReshapefeatures_18N_of_SCI/Reshape/shape:output:0*
T0*'
_output_shapes
:??????????
<National_Strategy_2_indicator/to_sparse_input/ignore_value/xConst*
_output_shapes
: *
dtype0*
valueB :
??????????
:National_Strategy_2_indicator/to_sparse_input/ignore_valueCastENational_Strategy_2_indicator/to_sparse_input/ignore_value/x:output:0*

DstT0	*

SrcT0*
_output_shapes
: ?
6National_Strategy_2_indicator/to_sparse_input/NotEqualNotEqualfeatures_19>National_Strategy_2_indicator/to_sparse_input/ignore_value:y:0*
T0	*'
_output_shapes
:??????????
5National_Strategy_2_indicator/to_sparse_input/indicesWhere:National_Strategy_2_indicator/to_sparse_input/NotEqual:z:0*'
_output_shapes
:??????????
4National_Strategy_2_indicator/to_sparse_input/valuesGatherNdfeatures_19=National_Strategy_2_indicator/to_sparse_input/indices:index:0*
Tindices0	*
Tparams0	*#
_output_shapes
:??????????
9National_Strategy_2_indicator/to_sparse_input/dense_shapeShapefeatures_19*
T0	*
_output_shapes
:*
out_type0	?
;National_Strategy_2_indicator/None_Lookup/LookupTableFindV2LookupTableFindV2Hnational_strategy_2_indicator_none_lookup_lookuptablefindv2_table_handle=National_Strategy_2_indicator/to_sparse_input/values:output:0Inational_strategy_2_indicator_none_lookup_lookuptablefindv2_default_value*	
Tin0	*

Tout0	*#
_output_shapes
:??????????
9National_Strategy_2_indicator/SparseToDense/default_valueConst*
_output_shapes
: *
dtype0	*
valueB	 R
??????????
+National_Strategy_2_indicator/SparseToDenseSparseToDense=National_Strategy_2_indicator/to_sparse_input/indices:index:0BNational_Strategy_2_indicator/to_sparse_input/dense_shape:output:0DNational_Strategy_2_indicator/None_Lookup/LookupTableFindV2:values:0BNational_Strategy_2_indicator/SparseToDense/default_value:output:0*
T0	*
Tindices0	*'
_output_shapes
:?????????p
+National_Strategy_2_indicator/one_hot/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??r
-National_Strategy_2_indicator/one_hot/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    m
+National_Strategy_2_indicator/one_hot/depthConst*
_output_shapes
: *
dtype0*
value	B :?
%National_Strategy_2_indicator/one_hotOneHot3National_Strategy_2_indicator/SparseToDense:dense:04National_Strategy_2_indicator/one_hot/depth:output:04National_Strategy_2_indicator/one_hot/Const:output:06National_Strategy_2_indicator/one_hot/Const_1:output:0*
T0*+
_output_shapes
:??????????
3National_Strategy_2_indicator/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
??????????
!National_Strategy_2_indicator/SumSum.National_Strategy_2_indicator/one_hot:output:0<National_Strategy_2_indicator/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:?????????}
#National_Strategy_2_indicator/ShapeShape*National_Strategy_2_indicator/Sum:output:0*
T0*
_output_shapes
:{
1National_Strategy_2_indicator/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: }
3National_Strategy_2_indicator/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:}
3National_Strategy_2_indicator/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
+National_Strategy_2_indicator/strided_sliceStridedSlice,National_Strategy_2_indicator/Shape:output:0:National_Strategy_2_indicator/strided_slice/stack:output:0<National_Strategy_2_indicator/strided_slice/stack_1:output:0<National_Strategy_2_indicator/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_masko
-National_Strategy_2_indicator/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
+National_Strategy_2_indicator/Reshape/shapePack4National_Strategy_2_indicator/strided_slice:output:06National_Strategy_2_indicator/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
%National_Strategy_2_indicator/ReshapeReshape*National_Strategy_2_indicator/Sum:output:04National_Strategy_2_indicator/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????{
0RnD_Org_indicator/to_sparse_input/ignore_value/xConst*
_output_shapes
: *
dtype0*
valueB :
??????????
.RnD_Org_indicator/to_sparse_input/ignore_valueCast9RnD_Org_indicator/to_sparse_input/ignore_value/x:output:0*

DstT0	*

SrcT0*
_output_shapes
: ?
*RnD_Org_indicator/to_sparse_input/NotEqualNotEqualfeatures_202RnD_Org_indicator/to_sparse_input/ignore_value:y:0*
T0	*'
_output_shapes
:??????????
)RnD_Org_indicator/to_sparse_input/indicesWhere.RnD_Org_indicator/to_sparse_input/NotEqual:z:0*'
_output_shapes
:??????????
(RnD_Org_indicator/to_sparse_input/valuesGatherNdfeatures_201RnD_Org_indicator/to_sparse_input/indices:index:0*
Tindices0	*
Tparams0	*#
_output_shapes
:?????????x
-RnD_Org_indicator/to_sparse_input/dense_shapeShapefeatures_20*
T0	*
_output_shapes
:*
out_type0	?
/RnD_Org_indicator/None_Lookup/LookupTableFindV2LookupTableFindV2<rnd_org_indicator_none_lookup_lookuptablefindv2_table_handle1RnD_Org_indicator/to_sparse_input/values:output:0=rnd_org_indicator_none_lookup_lookuptablefindv2_default_value*	
Tin0	*

Tout0	*#
_output_shapes
:?????????x
-RnD_Org_indicator/SparseToDense/default_valueConst*
_output_shapes
: *
dtype0	*
valueB	 R
??????????
RnD_Org_indicator/SparseToDenseSparseToDense1RnD_Org_indicator/to_sparse_input/indices:index:06RnD_Org_indicator/to_sparse_input/dense_shape:output:08RnD_Org_indicator/None_Lookup/LookupTableFindV2:values:06RnD_Org_indicator/SparseToDense/default_value:output:0*
T0	*
Tindices0	*'
_output_shapes
:?????????d
RnD_Org_indicator/one_hot/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??f
!RnD_Org_indicator/one_hot/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    a
RnD_Org_indicator/one_hot/depthConst*
_output_shapes
: *
dtype0*
value	B :?
RnD_Org_indicator/one_hotOneHot'RnD_Org_indicator/SparseToDense:dense:0(RnD_Org_indicator/one_hot/depth:output:0(RnD_Org_indicator/one_hot/Const:output:0*RnD_Org_indicator/one_hot/Const_1:output:0*
T0*+
_output_shapes
:?????????z
'RnD_Org_indicator/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
??????????
RnD_Org_indicator/SumSum"RnD_Org_indicator/one_hot:output:00RnD_Org_indicator/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:?????????e
RnD_Org_indicator/ShapeShapeRnD_Org_indicator/Sum:output:0*
T0*
_output_shapes
:o
%RnD_Org_indicator/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: q
'RnD_Org_indicator/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:q
'RnD_Org_indicator/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
RnD_Org_indicator/strided_sliceStridedSlice RnD_Org_indicator/Shape:output:0.RnD_Org_indicator/strided_slice/stack:output:00RnD_Org_indicator/strided_slice/stack_1:output:00RnD_Org_indicator/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskc
!RnD_Org_indicator/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
RnD_Org_indicator/Reshape/shapePack(RnD_Org_indicator/strided_slice:output:0*RnD_Org_indicator/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
RnD_Org_indicator/ReshapeReshapeRnD_Org_indicator/Sum:output:0(RnD_Org_indicator/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????}
2RnD_Stage_indicator/to_sparse_input/ignore_value/xConst*
_output_shapes
: *
dtype0*
valueB :
??????????
0RnD_Stage_indicator/to_sparse_input/ignore_valueCast;RnD_Stage_indicator/to_sparse_input/ignore_value/x:output:0*

DstT0	*

SrcT0*
_output_shapes
: ?
,RnD_Stage_indicator/to_sparse_input/NotEqualNotEqualfeatures_214RnD_Stage_indicator/to_sparse_input/ignore_value:y:0*
T0	*'
_output_shapes
:??????????
+RnD_Stage_indicator/to_sparse_input/indicesWhere0RnD_Stage_indicator/to_sparse_input/NotEqual:z:0*'
_output_shapes
:??????????
*RnD_Stage_indicator/to_sparse_input/valuesGatherNdfeatures_213RnD_Stage_indicator/to_sparse_input/indices:index:0*
Tindices0	*
Tparams0	*#
_output_shapes
:?????????z
/RnD_Stage_indicator/to_sparse_input/dense_shapeShapefeatures_21*
T0	*
_output_shapes
:*
out_type0	?
1RnD_Stage_indicator/None_Lookup/LookupTableFindV2LookupTableFindV2>rnd_stage_indicator_none_lookup_lookuptablefindv2_table_handle3RnD_Stage_indicator/to_sparse_input/values:output:0?rnd_stage_indicator_none_lookup_lookuptablefindv2_default_value*	
Tin0	*

Tout0	*#
_output_shapes
:?????????z
/RnD_Stage_indicator/SparseToDense/default_valueConst*
_output_shapes
: *
dtype0	*
valueB	 R
??????????
!RnD_Stage_indicator/SparseToDenseSparseToDense3RnD_Stage_indicator/to_sparse_input/indices:index:08RnD_Stage_indicator/to_sparse_input/dense_shape:output:0:RnD_Stage_indicator/None_Lookup/LookupTableFindV2:values:08RnD_Stage_indicator/SparseToDense/default_value:output:0*
T0	*
Tindices0	*'
_output_shapes
:?????????f
!RnD_Stage_indicator/one_hot/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??h
#RnD_Stage_indicator/one_hot/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    c
!RnD_Stage_indicator/one_hot/depthConst*
_output_shapes
: *
dtype0*
value	B :?
RnD_Stage_indicator/one_hotOneHot)RnD_Stage_indicator/SparseToDense:dense:0*RnD_Stage_indicator/one_hot/depth:output:0*RnD_Stage_indicator/one_hot/Const:output:0,RnD_Stage_indicator/one_hot/Const_1:output:0*
T0*+
_output_shapes
:?????????|
)RnD_Stage_indicator/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
??????????
RnD_Stage_indicator/SumSum$RnD_Stage_indicator/one_hot:output:02RnD_Stage_indicator/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:?????????i
RnD_Stage_indicator/ShapeShape RnD_Stage_indicator/Sum:output:0*
T0*
_output_shapes
:q
'RnD_Stage_indicator/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: s
)RnD_Stage_indicator/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:s
)RnD_Stage_indicator/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
!RnD_Stage_indicator/strided_sliceStridedSlice"RnD_Stage_indicator/Shape:output:00RnD_Stage_indicator/strided_slice/stack:output:02RnD_Stage_indicator/strided_slice/stack_1:output:02RnD_Stage_indicator/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maske
#RnD_Stage_indicator/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
!RnD_Stage_indicator/Reshape/shapePack*RnD_Stage_indicator/strided_slice:output:0,RnD_Stage_indicator/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
RnD_Stage_indicator/ReshapeReshape RnD_Stage_indicator/Sum:output:0*RnD_Stage_indicator/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????u
4STP_Code_11_indicator/to_sparse_input/ignore_value/xConst*
_output_shapes
: *
dtype0*
valueB B ?
.STP_Code_11_indicator/to_sparse_input/NotEqualNotEqualfeatures_22=STP_Code_11_indicator/to_sparse_input/ignore_value/x:output:0*
T0*'
_output_shapes
:??????????
-STP_Code_11_indicator/to_sparse_input/indicesWhere2STP_Code_11_indicator/to_sparse_input/NotEqual:z:0*'
_output_shapes
:??????????
,STP_Code_11_indicator/to_sparse_input/valuesGatherNdfeatures_225STP_Code_11_indicator/to_sparse_input/indices:index:0*
Tindices0	*
Tparams0*#
_output_shapes
:?????????|
1STP_Code_11_indicator/to_sparse_input/dense_shapeShapefeatures_22*
T0*
_output_shapes
:*
out_type0	?
3STP_Code_11_indicator/None_Lookup/LookupTableFindV2LookupTableFindV2@stp_code_11_indicator_none_lookup_lookuptablefindv2_table_handle5STP_Code_11_indicator/to_sparse_input/values:output:0Astp_code_11_indicator_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:?????????|
1STP_Code_11_indicator/SparseToDense/default_valueConst*
_output_shapes
: *
dtype0	*
valueB	 R
??????????
#STP_Code_11_indicator/SparseToDenseSparseToDense5STP_Code_11_indicator/to_sparse_input/indices:index:0:STP_Code_11_indicator/to_sparse_input/dense_shape:output:0<STP_Code_11_indicator/None_Lookup/LookupTableFindV2:values:0:STP_Code_11_indicator/SparseToDense/default_value:output:0*
T0	*
Tindices0	*'
_output_shapes
:?????????h
#STP_Code_11_indicator/one_hot/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??j
%STP_Code_11_indicator/one_hot/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    f
#STP_Code_11_indicator/one_hot/depthConst*
_output_shapes
: *
dtype0*
value
B :??
STP_Code_11_indicator/one_hotOneHot+STP_Code_11_indicator/SparseToDense:dense:0,STP_Code_11_indicator/one_hot/depth:output:0,STP_Code_11_indicator/one_hot/Const:output:0.STP_Code_11_indicator/one_hot/Const_1:output:0*
T0*,
_output_shapes
:??????????~
+STP_Code_11_indicator/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
??????????
STP_Code_11_indicator/SumSum&STP_Code_11_indicator/one_hot:output:04STP_Code_11_indicator/Sum/reduction_indices:output:0*
T0*(
_output_shapes
:??????????m
STP_Code_11_indicator/ShapeShape"STP_Code_11_indicator/Sum:output:0*
T0*
_output_shapes
:s
)STP_Code_11_indicator/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: u
+STP_Code_11_indicator/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+STP_Code_11_indicator/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
#STP_Code_11_indicator/strided_sliceStridedSlice$STP_Code_11_indicator/Shape:output:02STP_Code_11_indicator/strided_slice/stack:output:04STP_Code_11_indicator/strided_slice/stack_1:output:04STP_Code_11_indicator/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskh
%STP_Code_11_indicator/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value
B :??
#STP_Code_11_indicator/Reshape/shapePack,STP_Code_11_indicator/strided_slice:output:0.STP_Code_11_indicator/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
STP_Code_11_indicator/ReshapeReshape"STP_Code_11_indicator/Sum:output:0,STP_Code_11_indicator/Reshape/shape:output:0*
T0*(
_output_shapes
:??????????R
STP_Code_1_Weight/ShapeShapefeatures_23*
T0*
_output_shapes
:o
%STP_Code_1_Weight/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: q
'STP_Code_1_Weight/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:q
'STP_Code_1_Weight/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
STP_Code_1_Weight/strided_sliceStridedSlice STP_Code_1_Weight/Shape:output:0.STP_Code_1_Weight/strided_slice/stack:output:00STP_Code_1_Weight/strided_slice/stack_1:output:00STP_Code_1_Weight/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskc
!STP_Code_1_Weight/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
STP_Code_1_Weight/Reshape/shapePack(STP_Code_1_Weight/strided_slice:output:0*STP_Code_1_Weight/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
STP_Code_1_Weight/ReshapeReshapefeatures_23(STP_Code_1_Weight/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????u
4STP_Code_21_indicator/to_sparse_input/ignore_value/xConst*
_output_shapes
: *
dtype0*
valueB B ?
.STP_Code_21_indicator/to_sparse_input/NotEqualNotEqualfeatures_24=STP_Code_21_indicator/to_sparse_input/ignore_value/x:output:0*
T0*'
_output_shapes
:??????????
-STP_Code_21_indicator/to_sparse_input/indicesWhere2STP_Code_21_indicator/to_sparse_input/NotEqual:z:0*'
_output_shapes
:??????????
,STP_Code_21_indicator/to_sparse_input/valuesGatherNdfeatures_245STP_Code_21_indicator/to_sparse_input/indices:index:0*
Tindices0	*
Tparams0*#
_output_shapes
:?????????|
1STP_Code_21_indicator/to_sparse_input/dense_shapeShapefeatures_24*
T0*
_output_shapes
:*
out_type0	?
3STP_Code_21_indicator/None_Lookup/LookupTableFindV2LookupTableFindV2@stp_code_21_indicator_none_lookup_lookuptablefindv2_table_handle5STP_Code_21_indicator/to_sparse_input/values:output:0Astp_code_21_indicator_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:?????????|
1STP_Code_21_indicator/SparseToDense/default_valueConst*
_output_shapes
: *
dtype0	*
valueB	 R
??????????
#STP_Code_21_indicator/SparseToDenseSparseToDense5STP_Code_21_indicator/to_sparse_input/indices:index:0:STP_Code_21_indicator/to_sparse_input/dense_shape:output:0<STP_Code_21_indicator/None_Lookup/LookupTableFindV2:values:0:STP_Code_21_indicator/SparseToDense/default_value:output:0*
T0	*
Tindices0	*'
_output_shapes
:?????????h
#STP_Code_21_indicator/one_hot/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??j
%STP_Code_21_indicator/one_hot/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    f
#STP_Code_21_indicator/one_hot/depthConst*
_output_shapes
: *
dtype0*
value
B :??
STP_Code_21_indicator/one_hotOneHot+STP_Code_21_indicator/SparseToDense:dense:0,STP_Code_21_indicator/one_hot/depth:output:0,STP_Code_21_indicator/one_hot/Const:output:0.STP_Code_21_indicator/one_hot/Const_1:output:0*
T0*,
_output_shapes
:??????????~
+STP_Code_21_indicator/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
??????????
STP_Code_21_indicator/SumSum&STP_Code_21_indicator/one_hot:output:04STP_Code_21_indicator/Sum/reduction_indices:output:0*
T0*(
_output_shapes
:??????????m
STP_Code_21_indicator/ShapeShape"STP_Code_21_indicator/Sum:output:0*
T0*
_output_shapes
:s
)STP_Code_21_indicator/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: u
+STP_Code_21_indicator/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+STP_Code_21_indicator/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
#STP_Code_21_indicator/strided_sliceStridedSlice$STP_Code_21_indicator/Shape:output:02STP_Code_21_indicator/strided_slice/stack:output:04STP_Code_21_indicator/strided_slice/stack_1:output:04STP_Code_21_indicator/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskh
%STP_Code_21_indicator/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value
B :??
#STP_Code_21_indicator/Reshape/shapePack,STP_Code_21_indicator/strided_slice:output:0.STP_Code_21_indicator/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
STP_Code_21_indicator/ReshapeReshape"STP_Code_21_indicator/Sum:output:0,STP_Code_21_indicator/Reshape/shape:output:0*
T0*(
_output_shapes
:??????????R
STP_Code_2_Weight/ShapeShapefeatures_25*
T0*
_output_shapes
:o
%STP_Code_2_Weight/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: q
'STP_Code_2_Weight/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:q
'STP_Code_2_Weight/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
STP_Code_2_Weight/strided_sliceStridedSlice STP_Code_2_Weight/Shape:output:0.STP_Code_2_Weight/strided_slice/stack:output:00STP_Code_2_Weight/strided_slice/stack_1:output:00STP_Code_2_Weight/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskc
!STP_Code_2_Weight/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
STP_Code_2_Weight/Reshape/shapePack(STP_Code_2_Weight/strided_slice:output:0*STP_Code_2_Weight/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
STP_Code_2_Weight/ReshapeReshapefeatures_25(STP_Code_2_Weight/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????z
/SixT_2_indicator/to_sparse_input/ignore_value/xConst*
_output_shapes
: *
dtype0*
valueB :
??????????
-SixT_2_indicator/to_sparse_input/ignore_valueCast8SixT_2_indicator/to_sparse_input/ignore_value/x:output:0*

DstT0	*

SrcT0*
_output_shapes
: ?
)SixT_2_indicator/to_sparse_input/NotEqualNotEqualfeatures_261SixT_2_indicator/to_sparse_input/ignore_value:y:0*
T0	*'
_output_shapes
:??????????
(SixT_2_indicator/to_sparse_input/indicesWhere-SixT_2_indicator/to_sparse_input/NotEqual:z:0*'
_output_shapes
:??????????
'SixT_2_indicator/to_sparse_input/valuesGatherNdfeatures_260SixT_2_indicator/to_sparse_input/indices:index:0*
Tindices0	*
Tparams0	*#
_output_shapes
:?????????w
,SixT_2_indicator/to_sparse_input/dense_shapeShapefeatures_26*
T0	*
_output_shapes
:*
out_type0	?
.SixT_2_indicator/None_Lookup/LookupTableFindV2LookupTableFindV2;sixt_2_indicator_none_lookup_lookuptablefindv2_table_handle0SixT_2_indicator/to_sparse_input/values:output:0<sixt_2_indicator_none_lookup_lookuptablefindv2_default_value*	
Tin0	*

Tout0	*#
_output_shapes
:?????????w
,SixT_2_indicator/SparseToDense/default_valueConst*
_output_shapes
: *
dtype0	*
valueB	 R
??????????
SixT_2_indicator/SparseToDenseSparseToDense0SixT_2_indicator/to_sparse_input/indices:index:05SixT_2_indicator/to_sparse_input/dense_shape:output:07SixT_2_indicator/None_Lookup/LookupTableFindV2:values:05SixT_2_indicator/SparseToDense/default_value:output:0*
T0	*
Tindices0	*'
_output_shapes
:?????????c
SixT_2_indicator/one_hot/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??e
 SixT_2_indicator/one_hot/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    `
SixT_2_indicator/one_hot/depthConst*
_output_shapes
: *
dtype0*
value	B :?
SixT_2_indicator/one_hotOneHot&SixT_2_indicator/SparseToDense:dense:0'SixT_2_indicator/one_hot/depth:output:0'SixT_2_indicator/one_hot/Const:output:0)SixT_2_indicator/one_hot/Const_1:output:0*
T0*+
_output_shapes
:?????????y
&SixT_2_indicator/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
??????????
SixT_2_indicator/SumSum!SixT_2_indicator/one_hot:output:0/SixT_2_indicator/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:?????????c
SixT_2_indicator/ShapeShapeSixT_2_indicator/Sum:output:0*
T0*
_output_shapes
:n
$SixT_2_indicator/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: p
&SixT_2_indicator/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:p
&SixT_2_indicator/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
SixT_2_indicator/strided_sliceStridedSliceSixT_2_indicator/Shape:output:0-SixT_2_indicator/strided_slice/stack:output:0/SixT_2_indicator/strided_slice/stack_1:output:0/SixT_2_indicator/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskb
 SixT_2_indicator/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
SixT_2_indicator/Reshape/shapePack'SixT_2_indicator/strided_slice:output:0)SixT_2_indicator/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
SixT_2_indicator/ReshapeReshapeSixT_2_indicator/Sum:output:0'SixT_2_indicator/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????E

Year/ShapeShapefeatures_27*
T0*
_output_shapes
:b
Year/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: d
Year/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:d
Year/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
Year/strided_sliceStridedSliceYear/Shape:output:0!Year/strided_slice/stack:output:0#Year/strided_slice/stack_1:output:0#Year/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskV
Year/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
Year/Reshape/shapePackYear/strided_slice:output:0Year/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:s
Year/ReshapeReshapefeatures_27Year/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????V
concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
??????????
concatConcatV2*Application_Area_1_Weight/Reshape:output:0-Application_Area_1_indicator/Reshape:output:0*Application_Area_2_Weight/Reshape:output:0-Application_Area_2_indicator/Reshape:output:0(Cowork_Abroad_indicator/Reshape:output:0%Cowork_Cor_indicator/Reshape:output:0&Cowork_Inst_indicator/Reshape:output:0%Cowork_Uni_indicator/Reshape:output:0%Cowork_etc_indicator/Reshape:output:0&Econ_Social_indicator/Reshape:output:0%Green_Tech_indicator/Reshape:output:0Log_Duration/Reshape:output:0Log_RnD_Fund/Reshape:output:0%Multi_Year_indicator/Reshape:output:0N_Patent_App/Reshape:output:0N_Patent_Reg/Reshape:output:0#N_of_Korean_Patent/Reshape:output:0N_of_Paper/Reshape:output:0N_of_SCI/Reshape:output:0.National_Strategy_2_indicator/Reshape:output:0"RnD_Org_indicator/Reshape:output:0$RnD_Stage_indicator/Reshape:output:0&STP_Code_11_indicator/Reshape:output:0"STP_Code_1_Weight/Reshape:output:0&STP_Code_21_indicator/Reshape:output:0"STP_Code_2_Weight/Reshape:output:0!SixT_2_indicator/Reshape:output:0Year/Reshape:output:0concat/axis:output:0*
N*
T0*(
_output_shapes
:??????????_
IdentityIdentityconcat:output:0^NoOp*
T0*(
_output_shapes
:???????????
NoOpNoOp;^Application_Area_1_indicator/None_Lookup/LookupTableFindV2;^Application_Area_2_indicator/None_Lookup/LookupTableFindV26^Cowork_Abroad_indicator/None_Lookup/LookupTableFindV23^Cowork_Cor_indicator/None_Lookup/LookupTableFindV24^Cowork_Inst_indicator/None_Lookup/LookupTableFindV23^Cowork_Uni_indicator/None_Lookup/LookupTableFindV23^Cowork_etc_indicator/None_Lookup/LookupTableFindV24^Econ_Social_indicator/None_Lookup/LookupTableFindV23^Green_Tech_indicator/None_Lookup/LookupTableFindV23^Multi_Year_indicator/None_Lookup/LookupTableFindV2<^National_Strategy_2_indicator/None_Lookup/LookupTableFindV20^RnD_Org_indicator/None_Lookup/LookupTableFindV22^RnD_Stage_indicator/None_Lookup/LookupTableFindV24^STP_Code_11_indicator/None_Lookup/LookupTableFindV24^STP_Code_21_indicator/None_Lookup/LookupTableFindV2/^SixT_2_indicator/None_Lookup/LookupTableFindV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2x
:Application_Area_1_indicator/None_Lookup/LookupTableFindV2:Application_Area_1_indicator/None_Lookup/LookupTableFindV22x
:Application_Area_2_indicator/None_Lookup/LookupTableFindV2:Application_Area_2_indicator/None_Lookup/LookupTableFindV22n
5Cowork_Abroad_indicator/None_Lookup/LookupTableFindV25Cowork_Abroad_indicator/None_Lookup/LookupTableFindV22h
2Cowork_Cor_indicator/None_Lookup/LookupTableFindV22Cowork_Cor_indicator/None_Lookup/LookupTableFindV22j
3Cowork_Inst_indicator/None_Lookup/LookupTableFindV23Cowork_Inst_indicator/None_Lookup/LookupTableFindV22h
2Cowork_Uni_indicator/None_Lookup/LookupTableFindV22Cowork_Uni_indicator/None_Lookup/LookupTableFindV22h
2Cowork_etc_indicator/None_Lookup/LookupTableFindV22Cowork_etc_indicator/None_Lookup/LookupTableFindV22j
3Econ_Social_indicator/None_Lookup/LookupTableFindV23Econ_Social_indicator/None_Lookup/LookupTableFindV22h
2Green_Tech_indicator/None_Lookup/LookupTableFindV22Green_Tech_indicator/None_Lookup/LookupTableFindV22h
2Multi_Year_indicator/None_Lookup/LookupTableFindV22Multi_Year_indicator/None_Lookup/LookupTableFindV22z
;National_Strategy_2_indicator/None_Lookup/LookupTableFindV2;National_Strategy_2_indicator/None_Lookup/LookupTableFindV22b
/RnD_Org_indicator/None_Lookup/LookupTableFindV2/RnD_Org_indicator/None_Lookup/LookupTableFindV22f
1RnD_Stage_indicator/None_Lookup/LookupTableFindV21RnD_Stage_indicator/None_Lookup/LookupTableFindV22j
3STP_Code_11_indicator/None_Lookup/LookupTableFindV23STP_Code_11_indicator/None_Lookup/LookupTableFindV22j
3STP_Code_21_indicator/None_Lookup/LookupTableFindV23STP_Code_21_indicator/None_Lookup/LookupTableFindV22`
.SixT_2_indicator/None_Lookup/LookupTableFindV2.SixT_2_indicator/None_Lookup/LookupTableFindV2:Q M
'
_output_shapes
:?????????
"
_user_specified_name
features:QM
'
_output_shapes
:?????????
"
_user_specified_name
features:QM
'
_output_shapes
:?????????
"
_user_specified_name
features:QM
'
_output_shapes
:?????????
"
_user_specified_name
features:QM
'
_output_shapes
:?????????
"
_user_specified_name
features:QM
'
_output_shapes
:?????????
"
_user_specified_name
features:QM
'
_output_shapes
:?????????
"
_user_specified_name
features:QM
'
_output_shapes
:?????????
"
_user_specified_name
features:QM
'
_output_shapes
:?????????
"
_user_specified_name
features:Q	M
'
_output_shapes
:?????????
"
_user_specified_name
features:Q
M
'
_output_shapes
:?????????
"
_user_specified_name
features:QM
'
_output_shapes
:?????????
"
_user_specified_name
features:QM
'
_output_shapes
:?????????
"
_user_specified_name
features:QM
'
_output_shapes
:?????????
"
_user_specified_name
features:QM
'
_output_shapes
:?????????
"
_user_specified_name
features:QM
'
_output_shapes
:?????????
"
_user_specified_name
features:QM
'
_output_shapes
:?????????
"
_user_specified_name
features:QM
'
_output_shapes
:?????????
"
_user_specified_name
features:QM
'
_output_shapes
:?????????
"
_user_specified_name
features:QM
'
_output_shapes
:?????????
"
_user_specified_name
features:QM
'
_output_shapes
:?????????
"
_user_specified_name
features:QM
'
_output_shapes
:?????????
"
_user_specified_name
features:QM
'
_output_shapes
:?????????
"
_user_specified_name
features:QM
'
_output_shapes
:?????????
"
_user_specified_name
features:QM
'
_output_shapes
:?????????
"
_user_specified_name
features:QM
'
_output_shapes
:?????????
"
_user_specified_name
features:QM
'
_output_shapes
:?????????
"
_user_specified_name
features:QM
'
_output_shapes
:?????????
"
_user_specified_name
features:

_output_shapes
: :

_output_shapes
: :!

_output_shapes
: :#

_output_shapes
: :%

_output_shapes
: :'

_output_shapes
: :)

_output_shapes
: :+

_output_shapes
: :-

_output_shapes
: :/

_output_shapes
: :1

_output_shapes
: :3

_output_shapes
: :5

_output_shapes
: :7

_output_shapes
: :9

_output_shapes
: :;

_output_shapes
: 
??
?*
F__inference_sequential_layer_call_and_return_conditional_losses_132257
inputs_application_area_1$
 inputs_application_area_1_weight
inputs_application_area_2$
 inputs_application_area_2_weight
inputs_cowork_abroad
inputs_cowork_cor
inputs_cowork_inst
inputs_cowork_uni
inputs_cowork_etc
inputs_econ_social	
inputs_green_tech	
inputs_log_duration
inputs_log_rnd_fund
inputs_multi_year	
inputs_n_patent_app
inputs_n_patent_reg
inputs_n_of_korean_patent
inputs_n_of_paper
inputs_n_of_sci
inputs_national_strategy_2	
inputs_rnd_org	
inputs_rnd_stage	
inputs_stp_code_11
inputs_stp_code_1_weight
inputs_stp_code_21
inputs_stp_code_2_weight
inputs_sixt_2	
inputs_yearZ
Vdense_features_application_area_1_indicator_none_lookup_lookuptablefindv2_table_handle[
Wdense_features_application_area_1_indicator_none_lookup_lookuptablefindv2_default_value	Z
Vdense_features_application_area_2_indicator_none_lookup_lookuptablefindv2_table_handle[
Wdense_features_application_area_2_indicator_none_lookup_lookuptablefindv2_default_value	U
Qdense_features_cowork_abroad_indicator_none_lookup_lookuptablefindv2_table_handleV
Rdense_features_cowork_abroad_indicator_none_lookup_lookuptablefindv2_default_value	R
Ndense_features_cowork_cor_indicator_none_lookup_lookuptablefindv2_table_handleS
Odense_features_cowork_cor_indicator_none_lookup_lookuptablefindv2_default_value	S
Odense_features_cowork_inst_indicator_none_lookup_lookuptablefindv2_table_handleT
Pdense_features_cowork_inst_indicator_none_lookup_lookuptablefindv2_default_value	R
Ndense_features_cowork_uni_indicator_none_lookup_lookuptablefindv2_table_handleS
Odense_features_cowork_uni_indicator_none_lookup_lookuptablefindv2_default_value	R
Ndense_features_cowork_etc_indicator_none_lookup_lookuptablefindv2_table_handleS
Odense_features_cowork_etc_indicator_none_lookup_lookuptablefindv2_default_value	S
Odense_features_econ_social_indicator_none_lookup_lookuptablefindv2_table_handleT
Pdense_features_econ_social_indicator_none_lookup_lookuptablefindv2_default_value	R
Ndense_features_green_tech_indicator_none_lookup_lookuptablefindv2_table_handleS
Odense_features_green_tech_indicator_none_lookup_lookuptablefindv2_default_value	R
Ndense_features_multi_year_indicator_none_lookup_lookuptablefindv2_table_handleS
Odense_features_multi_year_indicator_none_lookup_lookuptablefindv2_default_value	[
Wdense_features_national_strategy_2_indicator_none_lookup_lookuptablefindv2_table_handle\
Xdense_features_national_strategy_2_indicator_none_lookup_lookuptablefindv2_default_value	O
Kdense_features_rnd_org_indicator_none_lookup_lookuptablefindv2_table_handleP
Ldense_features_rnd_org_indicator_none_lookup_lookuptablefindv2_default_value	Q
Mdense_features_rnd_stage_indicator_none_lookup_lookuptablefindv2_table_handleR
Ndense_features_rnd_stage_indicator_none_lookup_lookuptablefindv2_default_value	S
Odense_features_stp_code_11_indicator_none_lookup_lookuptablefindv2_table_handleT
Pdense_features_stp_code_11_indicator_none_lookup_lookuptablefindv2_default_value	S
Odense_features_stp_code_21_indicator_none_lookup_lookuptablefindv2_table_handleT
Pdense_features_stp_code_21_indicator_none_lookup_lookuptablefindv2_default_value	N
Jdense_features_sixt_2_indicator_none_lookup_lookuptablefindv2_table_handleO
Kdense_features_sixt_2_indicator_none_lookup_lookuptablefindv2_default_value	8
$dense_matmul_readvariableop_resource:
??4
%dense_biasadd_readvariableop_resource:	?:
&dense_1_matmul_readvariableop_resource:
??6
'dense_1_biasadd_readvariableop_resource:	?:
&dense_2_matmul_readvariableop_resource:
??6
'dense_2_biasadd_readvariableop_resource:	?9
&dense_3_matmul_readvariableop_resource:	?5
'dense_3_biasadd_readvariableop_resource:
identity??dense/BiasAdd/ReadVariableOp?dense/MatMul/ReadVariableOp?dense_1/BiasAdd/ReadVariableOp?dense_1/MatMul/ReadVariableOp?dense_2/BiasAdd/ReadVariableOp?dense_2/MatMul/ReadVariableOp?dense_3/BiasAdd/ReadVariableOp?dense_3/MatMul/ReadVariableOp?Idense_features/Application_Area_1_indicator/None_Lookup/LookupTableFindV2?Idense_features/Application_Area_2_indicator/None_Lookup/LookupTableFindV2?Ddense_features/Cowork_Abroad_indicator/None_Lookup/LookupTableFindV2?Adense_features/Cowork_Cor_indicator/None_Lookup/LookupTableFindV2?Bdense_features/Cowork_Inst_indicator/None_Lookup/LookupTableFindV2?Adense_features/Cowork_Uni_indicator/None_Lookup/LookupTableFindV2?Adense_features/Cowork_etc_indicator/None_Lookup/LookupTableFindV2?Bdense_features/Econ_Social_indicator/None_Lookup/LookupTableFindV2?Adense_features/Green_Tech_indicator/None_Lookup/LookupTableFindV2?Adense_features/Multi_Year_indicator/None_Lookup/LookupTableFindV2?Jdense_features/National_Strategy_2_indicator/None_Lookup/LookupTableFindV2?>dense_features/RnD_Org_indicator/None_Lookup/LookupTableFindV2?@dense_features/RnD_Stage_indicator/None_Lookup/LookupTableFindV2?Bdense_features/STP_Code_11_indicator/None_Lookup/LookupTableFindV2?Bdense_features/STP_Code_21_indicator/None_Lookup/LookupTableFindV2?=dense_features/SixT_2_indicator/None_Lookup/LookupTableFindV2~
.dense_features/Application_Area_1_Weight/ShapeShape inputs_application_area_1_weight*
T0*
_output_shapes
:?
<dense_features/Application_Area_1_Weight/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
>dense_features/Application_Area_1_Weight/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
>dense_features/Application_Area_1_Weight/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
6dense_features/Application_Area_1_Weight/strided_sliceStridedSlice7dense_features/Application_Area_1_Weight/Shape:output:0Edense_features/Application_Area_1_Weight/strided_slice/stack:output:0Gdense_features/Application_Area_1_Weight/strided_slice/stack_1:output:0Gdense_features/Application_Area_1_Weight/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskz
8dense_features/Application_Area_1_Weight/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
6dense_features/Application_Area_1_Weight/Reshape/shapePack?dense_features/Application_Area_1_Weight/strided_slice:output:0Adense_features/Application_Area_1_Weight/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
0dense_features/Application_Area_1_Weight/ReshapeReshape inputs_application_area_1_weight?dense_features/Application_Area_1_Weight/Reshape/shape:output:0*
T0*'
_output_shapes
:??????????
Jdense_features/Application_Area_1_indicator/to_sparse_input/ignore_value/xConst*
_output_shapes
: *
dtype0*
valueB B ?
Ddense_features/Application_Area_1_indicator/to_sparse_input/NotEqualNotEqualinputs_application_area_1Sdense_features/Application_Area_1_indicator/to_sparse_input/ignore_value/x:output:0*
T0*'
_output_shapes
:??????????
Cdense_features/Application_Area_1_indicator/to_sparse_input/indicesWhereHdense_features/Application_Area_1_indicator/to_sparse_input/NotEqual:z:0*'
_output_shapes
:??????????
Bdense_features/Application_Area_1_indicator/to_sparse_input/valuesGatherNdinputs_application_area_1Kdense_features/Application_Area_1_indicator/to_sparse_input/indices:index:0*
Tindices0	*
Tparams0*#
_output_shapes
:??????????
Gdense_features/Application_Area_1_indicator/to_sparse_input/dense_shapeShapeinputs_application_area_1*
T0*
_output_shapes
:*
out_type0	?
Idense_features/Application_Area_1_indicator/None_Lookup/LookupTableFindV2LookupTableFindV2Vdense_features_application_area_1_indicator_none_lookup_lookuptablefindv2_table_handleKdense_features/Application_Area_1_indicator/to_sparse_input/values:output:0Wdense_features_application_area_1_indicator_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:??????????
Gdense_features/Application_Area_1_indicator/SparseToDense/default_valueConst*
_output_shapes
: *
dtype0	*
valueB	 R
??????????
9dense_features/Application_Area_1_indicator/SparseToDenseSparseToDenseKdense_features/Application_Area_1_indicator/to_sparse_input/indices:index:0Pdense_features/Application_Area_1_indicator/to_sparse_input/dense_shape:output:0Rdense_features/Application_Area_1_indicator/None_Lookup/LookupTableFindV2:values:0Pdense_features/Application_Area_1_indicator/SparseToDense/default_value:output:0*
T0	*
Tindices0	*'
_output_shapes
:?????????~
9dense_features/Application_Area_1_indicator/one_hot/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
;dense_features/Application_Area_1_indicator/one_hot/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    {
9dense_features/Application_Area_1_indicator/one_hot/depthConst*
_output_shapes
: *
dtype0*
value	B :!?
3dense_features/Application_Area_1_indicator/one_hotOneHotAdense_features/Application_Area_1_indicator/SparseToDense:dense:0Bdense_features/Application_Area_1_indicator/one_hot/depth:output:0Bdense_features/Application_Area_1_indicator/one_hot/Const:output:0Ddense_features/Application_Area_1_indicator/one_hot/Const_1:output:0*
T0*+
_output_shapes
:?????????!?
Adense_features/Application_Area_1_indicator/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
??????????
/dense_features/Application_Area_1_indicator/SumSum<dense_features/Application_Area_1_indicator/one_hot:output:0Jdense_features/Application_Area_1_indicator/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:?????????!?
1dense_features/Application_Area_1_indicator/ShapeShape8dense_features/Application_Area_1_indicator/Sum:output:0*
T0*
_output_shapes
:?
?dense_features/Application_Area_1_indicator/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
Adense_features/Application_Area_1_indicator/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
Adense_features/Application_Area_1_indicator/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
9dense_features/Application_Area_1_indicator/strided_sliceStridedSlice:dense_features/Application_Area_1_indicator/Shape:output:0Hdense_features/Application_Area_1_indicator/strided_slice/stack:output:0Jdense_features/Application_Area_1_indicator/strided_slice/stack_1:output:0Jdense_features/Application_Area_1_indicator/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask}
;dense_features/Application_Area_1_indicator/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :!?
9dense_features/Application_Area_1_indicator/Reshape/shapePackBdense_features/Application_Area_1_indicator/strided_slice:output:0Ddense_features/Application_Area_1_indicator/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
3dense_features/Application_Area_1_indicator/ReshapeReshape8dense_features/Application_Area_1_indicator/Sum:output:0Bdense_features/Application_Area_1_indicator/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????!~
.dense_features/Application_Area_2_Weight/ShapeShape inputs_application_area_2_weight*
T0*
_output_shapes
:?
<dense_features/Application_Area_2_Weight/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
>dense_features/Application_Area_2_Weight/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
>dense_features/Application_Area_2_Weight/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
6dense_features/Application_Area_2_Weight/strided_sliceStridedSlice7dense_features/Application_Area_2_Weight/Shape:output:0Edense_features/Application_Area_2_Weight/strided_slice/stack:output:0Gdense_features/Application_Area_2_Weight/strided_slice/stack_1:output:0Gdense_features/Application_Area_2_Weight/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskz
8dense_features/Application_Area_2_Weight/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
6dense_features/Application_Area_2_Weight/Reshape/shapePack?dense_features/Application_Area_2_Weight/strided_slice:output:0Adense_features/Application_Area_2_Weight/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
0dense_features/Application_Area_2_Weight/ReshapeReshape inputs_application_area_2_weight?dense_features/Application_Area_2_Weight/Reshape/shape:output:0*
T0*'
_output_shapes
:??????????
Jdense_features/Application_Area_2_indicator/to_sparse_input/ignore_value/xConst*
_output_shapes
: *
dtype0*
valueB B ?
Ddense_features/Application_Area_2_indicator/to_sparse_input/NotEqualNotEqualinputs_application_area_2Sdense_features/Application_Area_2_indicator/to_sparse_input/ignore_value/x:output:0*
T0*'
_output_shapes
:??????????
Cdense_features/Application_Area_2_indicator/to_sparse_input/indicesWhereHdense_features/Application_Area_2_indicator/to_sparse_input/NotEqual:z:0*'
_output_shapes
:??????????
Bdense_features/Application_Area_2_indicator/to_sparse_input/valuesGatherNdinputs_application_area_2Kdense_features/Application_Area_2_indicator/to_sparse_input/indices:index:0*
Tindices0	*
Tparams0*#
_output_shapes
:??????????
Gdense_features/Application_Area_2_indicator/to_sparse_input/dense_shapeShapeinputs_application_area_2*
T0*
_output_shapes
:*
out_type0	?
Idense_features/Application_Area_2_indicator/None_Lookup/LookupTableFindV2LookupTableFindV2Vdense_features_application_area_2_indicator_none_lookup_lookuptablefindv2_table_handleKdense_features/Application_Area_2_indicator/to_sparse_input/values:output:0Wdense_features_application_area_2_indicator_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:??????????
Gdense_features/Application_Area_2_indicator/SparseToDense/default_valueConst*
_output_shapes
: *
dtype0	*
valueB	 R
??????????
9dense_features/Application_Area_2_indicator/SparseToDenseSparseToDenseKdense_features/Application_Area_2_indicator/to_sparse_input/indices:index:0Pdense_features/Application_Area_2_indicator/to_sparse_input/dense_shape:output:0Rdense_features/Application_Area_2_indicator/None_Lookup/LookupTableFindV2:values:0Pdense_features/Application_Area_2_indicator/SparseToDense/default_value:output:0*
T0	*
Tindices0	*'
_output_shapes
:?????????~
9dense_features/Application_Area_2_indicator/one_hot/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
;dense_features/Application_Area_2_indicator/one_hot/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    {
9dense_features/Application_Area_2_indicator/one_hot/depthConst*
_output_shapes
: *
dtype0*
value	B :"?
3dense_features/Application_Area_2_indicator/one_hotOneHotAdense_features/Application_Area_2_indicator/SparseToDense:dense:0Bdense_features/Application_Area_2_indicator/one_hot/depth:output:0Bdense_features/Application_Area_2_indicator/one_hot/Const:output:0Ddense_features/Application_Area_2_indicator/one_hot/Const_1:output:0*
T0*+
_output_shapes
:?????????"?
Adense_features/Application_Area_2_indicator/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
??????????
/dense_features/Application_Area_2_indicator/SumSum<dense_features/Application_Area_2_indicator/one_hot:output:0Jdense_features/Application_Area_2_indicator/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:?????????"?
1dense_features/Application_Area_2_indicator/ShapeShape8dense_features/Application_Area_2_indicator/Sum:output:0*
T0*
_output_shapes
:?
?dense_features/Application_Area_2_indicator/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
Adense_features/Application_Area_2_indicator/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
Adense_features/Application_Area_2_indicator/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
9dense_features/Application_Area_2_indicator/strided_sliceStridedSlice:dense_features/Application_Area_2_indicator/Shape:output:0Hdense_features/Application_Area_2_indicator/strided_slice/stack:output:0Jdense_features/Application_Area_2_indicator/strided_slice/stack_1:output:0Jdense_features/Application_Area_2_indicator/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask}
;dense_features/Application_Area_2_indicator/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :"?
9dense_features/Application_Area_2_indicator/Reshape/shapePackBdense_features/Application_Area_2_indicator/strided_slice:output:0Ddense_features/Application_Area_2_indicator/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
3dense_features/Application_Area_2_indicator/ReshapeReshape8dense_features/Application_Area_2_indicator/Sum:output:0Bdense_features/Application_Area_2_indicator/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????"?
Edense_features/Cowork_Abroad_indicator/to_sparse_input/ignore_value/xConst*
_output_shapes
: *
dtype0*
valueB B ?
?dense_features/Cowork_Abroad_indicator/to_sparse_input/NotEqualNotEqualinputs_cowork_abroadNdense_features/Cowork_Abroad_indicator/to_sparse_input/ignore_value/x:output:0*
T0*'
_output_shapes
:??????????
>dense_features/Cowork_Abroad_indicator/to_sparse_input/indicesWhereCdense_features/Cowork_Abroad_indicator/to_sparse_input/NotEqual:z:0*'
_output_shapes
:??????????
=dense_features/Cowork_Abroad_indicator/to_sparse_input/valuesGatherNdinputs_cowork_abroadFdense_features/Cowork_Abroad_indicator/to_sparse_input/indices:index:0*
Tindices0	*
Tparams0*#
_output_shapes
:??????????
Bdense_features/Cowork_Abroad_indicator/to_sparse_input/dense_shapeShapeinputs_cowork_abroad*
T0*
_output_shapes
:*
out_type0	?
Ddense_features/Cowork_Abroad_indicator/None_Lookup/LookupTableFindV2LookupTableFindV2Qdense_features_cowork_abroad_indicator_none_lookup_lookuptablefindv2_table_handleFdense_features/Cowork_Abroad_indicator/to_sparse_input/values:output:0Rdense_features_cowork_abroad_indicator_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:??????????
Bdense_features/Cowork_Abroad_indicator/SparseToDense/default_valueConst*
_output_shapes
: *
dtype0	*
valueB	 R
??????????
4dense_features/Cowork_Abroad_indicator/SparseToDenseSparseToDenseFdense_features/Cowork_Abroad_indicator/to_sparse_input/indices:index:0Kdense_features/Cowork_Abroad_indicator/to_sparse_input/dense_shape:output:0Mdense_features/Cowork_Abroad_indicator/None_Lookup/LookupTableFindV2:values:0Kdense_features/Cowork_Abroad_indicator/SparseToDense/default_value:output:0*
T0	*
Tindices0	*'
_output_shapes
:?????????y
4dense_features/Cowork_Abroad_indicator/one_hot/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??{
6dense_features/Cowork_Abroad_indicator/one_hot/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    v
4dense_features/Cowork_Abroad_indicator/one_hot/depthConst*
_output_shapes
: *
dtype0*
value	B :?
.dense_features/Cowork_Abroad_indicator/one_hotOneHot<dense_features/Cowork_Abroad_indicator/SparseToDense:dense:0=dense_features/Cowork_Abroad_indicator/one_hot/depth:output:0=dense_features/Cowork_Abroad_indicator/one_hot/Const:output:0?dense_features/Cowork_Abroad_indicator/one_hot/Const_1:output:0*
T0*+
_output_shapes
:??????????
<dense_features/Cowork_Abroad_indicator/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
??????????
*dense_features/Cowork_Abroad_indicator/SumSum7dense_features/Cowork_Abroad_indicator/one_hot:output:0Edense_features/Cowork_Abroad_indicator/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:??????????
,dense_features/Cowork_Abroad_indicator/ShapeShape3dense_features/Cowork_Abroad_indicator/Sum:output:0*
T0*
_output_shapes
:?
:dense_features/Cowork_Abroad_indicator/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
<dense_features/Cowork_Abroad_indicator/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
<dense_features/Cowork_Abroad_indicator/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
4dense_features/Cowork_Abroad_indicator/strided_sliceStridedSlice5dense_features/Cowork_Abroad_indicator/Shape:output:0Cdense_features/Cowork_Abroad_indicator/strided_slice/stack:output:0Edense_features/Cowork_Abroad_indicator/strided_slice/stack_1:output:0Edense_features/Cowork_Abroad_indicator/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskx
6dense_features/Cowork_Abroad_indicator/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
4dense_features/Cowork_Abroad_indicator/Reshape/shapePack=dense_features/Cowork_Abroad_indicator/strided_slice:output:0?dense_features/Cowork_Abroad_indicator/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
.dense_features/Cowork_Abroad_indicator/ReshapeReshape3dense_features/Cowork_Abroad_indicator/Sum:output:0=dense_features/Cowork_Abroad_indicator/Reshape/shape:output:0*
T0*'
_output_shapes
:??????????
Bdense_features/Cowork_Cor_indicator/to_sparse_input/ignore_value/xConst*
_output_shapes
: *
dtype0*
valueB B ?
<dense_features/Cowork_Cor_indicator/to_sparse_input/NotEqualNotEqualinputs_cowork_corKdense_features/Cowork_Cor_indicator/to_sparse_input/ignore_value/x:output:0*
T0*'
_output_shapes
:??????????
;dense_features/Cowork_Cor_indicator/to_sparse_input/indicesWhere@dense_features/Cowork_Cor_indicator/to_sparse_input/NotEqual:z:0*'
_output_shapes
:??????????
:dense_features/Cowork_Cor_indicator/to_sparse_input/valuesGatherNdinputs_cowork_corCdense_features/Cowork_Cor_indicator/to_sparse_input/indices:index:0*
Tindices0	*
Tparams0*#
_output_shapes
:??????????
?dense_features/Cowork_Cor_indicator/to_sparse_input/dense_shapeShapeinputs_cowork_cor*
T0*
_output_shapes
:*
out_type0	?
Adense_features/Cowork_Cor_indicator/None_Lookup/LookupTableFindV2LookupTableFindV2Ndense_features_cowork_cor_indicator_none_lookup_lookuptablefindv2_table_handleCdense_features/Cowork_Cor_indicator/to_sparse_input/values:output:0Odense_features_cowork_cor_indicator_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:??????????
?dense_features/Cowork_Cor_indicator/SparseToDense/default_valueConst*
_output_shapes
: *
dtype0	*
valueB	 R
??????????
1dense_features/Cowork_Cor_indicator/SparseToDenseSparseToDenseCdense_features/Cowork_Cor_indicator/to_sparse_input/indices:index:0Hdense_features/Cowork_Cor_indicator/to_sparse_input/dense_shape:output:0Jdense_features/Cowork_Cor_indicator/None_Lookup/LookupTableFindV2:values:0Hdense_features/Cowork_Cor_indicator/SparseToDense/default_value:output:0*
T0	*
Tindices0	*'
_output_shapes
:?????????v
1dense_features/Cowork_Cor_indicator/one_hot/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??x
3dense_features/Cowork_Cor_indicator/one_hot/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    s
1dense_features/Cowork_Cor_indicator/one_hot/depthConst*
_output_shapes
: *
dtype0*
value	B :?
+dense_features/Cowork_Cor_indicator/one_hotOneHot9dense_features/Cowork_Cor_indicator/SparseToDense:dense:0:dense_features/Cowork_Cor_indicator/one_hot/depth:output:0:dense_features/Cowork_Cor_indicator/one_hot/Const:output:0<dense_features/Cowork_Cor_indicator/one_hot/Const_1:output:0*
T0*+
_output_shapes
:??????????
9dense_features/Cowork_Cor_indicator/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
??????????
'dense_features/Cowork_Cor_indicator/SumSum4dense_features/Cowork_Cor_indicator/one_hot:output:0Bdense_features/Cowork_Cor_indicator/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:??????????
)dense_features/Cowork_Cor_indicator/ShapeShape0dense_features/Cowork_Cor_indicator/Sum:output:0*
T0*
_output_shapes
:?
7dense_features/Cowork_Cor_indicator/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
9dense_features/Cowork_Cor_indicator/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
9dense_features/Cowork_Cor_indicator/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
1dense_features/Cowork_Cor_indicator/strided_sliceStridedSlice2dense_features/Cowork_Cor_indicator/Shape:output:0@dense_features/Cowork_Cor_indicator/strided_slice/stack:output:0Bdense_features/Cowork_Cor_indicator/strided_slice/stack_1:output:0Bdense_features/Cowork_Cor_indicator/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_masku
3dense_features/Cowork_Cor_indicator/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
1dense_features/Cowork_Cor_indicator/Reshape/shapePack:dense_features/Cowork_Cor_indicator/strided_slice:output:0<dense_features/Cowork_Cor_indicator/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
+dense_features/Cowork_Cor_indicator/ReshapeReshape0dense_features/Cowork_Cor_indicator/Sum:output:0:dense_features/Cowork_Cor_indicator/Reshape/shape:output:0*
T0*'
_output_shapes
:??????????
Cdense_features/Cowork_Inst_indicator/to_sparse_input/ignore_value/xConst*
_output_shapes
: *
dtype0*
valueB B ?
=dense_features/Cowork_Inst_indicator/to_sparse_input/NotEqualNotEqualinputs_cowork_instLdense_features/Cowork_Inst_indicator/to_sparse_input/ignore_value/x:output:0*
T0*'
_output_shapes
:??????????
<dense_features/Cowork_Inst_indicator/to_sparse_input/indicesWhereAdense_features/Cowork_Inst_indicator/to_sparse_input/NotEqual:z:0*'
_output_shapes
:??????????
;dense_features/Cowork_Inst_indicator/to_sparse_input/valuesGatherNdinputs_cowork_instDdense_features/Cowork_Inst_indicator/to_sparse_input/indices:index:0*
Tindices0	*
Tparams0*#
_output_shapes
:??????????
@dense_features/Cowork_Inst_indicator/to_sparse_input/dense_shapeShapeinputs_cowork_inst*
T0*
_output_shapes
:*
out_type0	?
Bdense_features/Cowork_Inst_indicator/None_Lookup/LookupTableFindV2LookupTableFindV2Odense_features_cowork_inst_indicator_none_lookup_lookuptablefindv2_table_handleDdense_features/Cowork_Inst_indicator/to_sparse_input/values:output:0Pdense_features_cowork_inst_indicator_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:??????????
@dense_features/Cowork_Inst_indicator/SparseToDense/default_valueConst*
_output_shapes
: *
dtype0	*
valueB	 R
??????????
2dense_features/Cowork_Inst_indicator/SparseToDenseSparseToDenseDdense_features/Cowork_Inst_indicator/to_sparse_input/indices:index:0Idense_features/Cowork_Inst_indicator/to_sparse_input/dense_shape:output:0Kdense_features/Cowork_Inst_indicator/None_Lookup/LookupTableFindV2:values:0Idense_features/Cowork_Inst_indicator/SparseToDense/default_value:output:0*
T0	*
Tindices0	*'
_output_shapes
:?????????w
2dense_features/Cowork_Inst_indicator/one_hot/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??y
4dense_features/Cowork_Inst_indicator/one_hot/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    t
2dense_features/Cowork_Inst_indicator/one_hot/depthConst*
_output_shapes
: *
dtype0*
value	B :?
,dense_features/Cowork_Inst_indicator/one_hotOneHot:dense_features/Cowork_Inst_indicator/SparseToDense:dense:0;dense_features/Cowork_Inst_indicator/one_hot/depth:output:0;dense_features/Cowork_Inst_indicator/one_hot/Const:output:0=dense_features/Cowork_Inst_indicator/one_hot/Const_1:output:0*
T0*+
_output_shapes
:??????????
:dense_features/Cowork_Inst_indicator/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
??????????
(dense_features/Cowork_Inst_indicator/SumSum5dense_features/Cowork_Inst_indicator/one_hot:output:0Cdense_features/Cowork_Inst_indicator/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:??????????
*dense_features/Cowork_Inst_indicator/ShapeShape1dense_features/Cowork_Inst_indicator/Sum:output:0*
T0*
_output_shapes
:?
8dense_features/Cowork_Inst_indicator/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
:dense_features/Cowork_Inst_indicator/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
:dense_features/Cowork_Inst_indicator/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
2dense_features/Cowork_Inst_indicator/strided_sliceStridedSlice3dense_features/Cowork_Inst_indicator/Shape:output:0Adense_features/Cowork_Inst_indicator/strided_slice/stack:output:0Cdense_features/Cowork_Inst_indicator/strided_slice/stack_1:output:0Cdense_features/Cowork_Inst_indicator/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskv
4dense_features/Cowork_Inst_indicator/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
2dense_features/Cowork_Inst_indicator/Reshape/shapePack;dense_features/Cowork_Inst_indicator/strided_slice:output:0=dense_features/Cowork_Inst_indicator/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
,dense_features/Cowork_Inst_indicator/ReshapeReshape1dense_features/Cowork_Inst_indicator/Sum:output:0;dense_features/Cowork_Inst_indicator/Reshape/shape:output:0*
T0*'
_output_shapes
:??????????
Bdense_features/Cowork_Uni_indicator/to_sparse_input/ignore_value/xConst*
_output_shapes
: *
dtype0*
valueB B ?
<dense_features/Cowork_Uni_indicator/to_sparse_input/NotEqualNotEqualinputs_cowork_uniKdense_features/Cowork_Uni_indicator/to_sparse_input/ignore_value/x:output:0*
T0*'
_output_shapes
:??????????
;dense_features/Cowork_Uni_indicator/to_sparse_input/indicesWhere@dense_features/Cowork_Uni_indicator/to_sparse_input/NotEqual:z:0*'
_output_shapes
:??????????
:dense_features/Cowork_Uni_indicator/to_sparse_input/valuesGatherNdinputs_cowork_uniCdense_features/Cowork_Uni_indicator/to_sparse_input/indices:index:0*
Tindices0	*
Tparams0*#
_output_shapes
:??????????
?dense_features/Cowork_Uni_indicator/to_sparse_input/dense_shapeShapeinputs_cowork_uni*
T0*
_output_shapes
:*
out_type0	?
Adense_features/Cowork_Uni_indicator/None_Lookup/LookupTableFindV2LookupTableFindV2Ndense_features_cowork_uni_indicator_none_lookup_lookuptablefindv2_table_handleCdense_features/Cowork_Uni_indicator/to_sparse_input/values:output:0Odense_features_cowork_uni_indicator_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:??????????
?dense_features/Cowork_Uni_indicator/SparseToDense/default_valueConst*
_output_shapes
: *
dtype0	*
valueB	 R
??????????
1dense_features/Cowork_Uni_indicator/SparseToDenseSparseToDenseCdense_features/Cowork_Uni_indicator/to_sparse_input/indices:index:0Hdense_features/Cowork_Uni_indicator/to_sparse_input/dense_shape:output:0Jdense_features/Cowork_Uni_indicator/None_Lookup/LookupTableFindV2:values:0Hdense_features/Cowork_Uni_indicator/SparseToDense/default_value:output:0*
T0	*
Tindices0	*'
_output_shapes
:?????????v
1dense_features/Cowork_Uni_indicator/one_hot/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??x
3dense_features/Cowork_Uni_indicator/one_hot/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    s
1dense_features/Cowork_Uni_indicator/one_hot/depthConst*
_output_shapes
: *
dtype0*
value	B :?
+dense_features/Cowork_Uni_indicator/one_hotOneHot9dense_features/Cowork_Uni_indicator/SparseToDense:dense:0:dense_features/Cowork_Uni_indicator/one_hot/depth:output:0:dense_features/Cowork_Uni_indicator/one_hot/Const:output:0<dense_features/Cowork_Uni_indicator/one_hot/Const_1:output:0*
T0*+
_output_shapes
:??????????
9dense_features/Cowork_Uni_indicator/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
??????????
'dense_features/Cowork_Uni_indicator/SumSum4dense_features/Cowork_Uni_indicator/one_hot:output:0Bdense_features/Cowork_Uni_indicator/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:??????????
)dense_features/Cowork_Uni_indicator/ShapeShape0dense_features/Cowork_Uni_indicator/Sum:output:0*
T0*
_output_shapes
:?
7dense_features/Cowork_Uni_indicator/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
9dense_features/Cowork_Uni_indicator/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
9dense_features/Cowork_Uni_indicator/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
1dense_features/Cowork_Uni_indicator/strided_sliceStridedSlice2dense_features/Cowork_Uni_indicator/Shape:output:0@dense_features/Cowork_Uni_indicator/strided_slice/stack:output:0Bdense_features/Cowork_Uni_indicator/strided_slice/stack_1:output:0Bdense_features/Cowork_Uni_indicator/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_masku
3dense_features/Cowork_Uni_indicator/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
1dense_features/Cowork_Uni_indicator/Reshape/shapePack:dense_features/Cowork_Uni_indicator/strided_slice:output:0<dense_features/Cowork_Uni_indicator/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
+dense_features/Cowork_Uni_indicator/ReshapeReshape0dense_features/Cowork_Uni_indicator/Sum:output:0:dense_features/Cowork_Uni_indicator/Reshape/shape:output:0*
T0*'
_output_shapes
:??????????
Bdense_features/Cowork_etc_indicator/to_sparse_input/ignore_value/xConst*
_output_shapes
: *
dtype0*
valueB B ?
<dense_features/Cowork_etc_indicator/to_sparse_input/NotEqualNotEqualinputs_cowork_etcKdense_features/Cowork_etc_indicator/to_sparse_input/ignore_value/x:output:0*
T0*'
_output_shapes
:??????????
;dense_features/Cowork_etc_indicator/to_sparse_input/indicesWhere@dense_features/Cowork_etc_indicator/to_sparse_input/NotEqual:z:0*'
_output_shapes
:??????????
:dense_features/Cowork_etc_indicator/to_sparse_input/valuesGatherNdinputs_cowork_etcCdense_features/Cowork_etc_indicator/to_sparse_input/indices:index:0*
Tindices0	*
Tparams0*#
_output_shapes
:??????????
?dense_features/Cowork_etc_indicator/to_sparse_input/dense_shapeShapeinputs_cowork_etc*
T0*
_output_shapes
:*
out_type0	?
Adense_features/Cowork_etc_indicator/None_Lookup/LookupTableFindV2LookupTableFindV2Ndense_features_cowork_etc_indicator_none_lookup_lookuptablefindv2_table_handleCdense_features/Cowork_etc_indicator/to_sparse_input/values:output:0Odense_features_cowork_etc_indicator_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:??????????
?dense_features/Cowork_etc_indicator/SparseToDense/default_valueConst*
_output_shapes
: *
dtype0	*
valueB	 R
??????????
1dense_features/Cowork_etc_indicator/SparseToDenseSparseToDenseCdense_features/Cowork_etc_indicator/to_sparse_input/indices:index:0Hdense_features/Cowork_etc_indicator/to_sparse_input/dense_shape:output:0Jdense_features/Cowork_etc_indicator/None_Lookup/LookupTableFindV2:values:0Hdense_features/Cowork_etc_indicator/SparseToDense/default_value:output:0*
T0	*
Tindices0	*'
_output_shapes
:?????????v
1dense_features/Cowork_etc_indicator/one_hot/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??x
3dense_features/Cowork_etc_indicator/one_hot/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    s
1dense_features/Cowork_etc_indicator/one_hot/depthConst*
_output_shapes
: *
dtype0*
value	B :?
+dense_features/Cowork_etc_indicator/one_hotOneHot9dense_features/Cowork_etc_indicator/SparseToDense:dense:0:dense_features/Cowork_etc_indicator/one_hot/depth:output:0:dense_features/Cowork_etc_indicator/one_hot/Const:output:0<dense_features/Cowork_etc_indicator/one_hot/Const_1:output:0*
T0*+
_output_shapes
:??????????
9dense_features/Cowork_etc_indicator/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
??????????
'dense_features/Cowork_etc_indicator/SumSum4dense_features/Cowork_etc_indicator/one_hot:output:0Bdense_features/Cowork_etc_indicator/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:??????????
)dense_features/Cowork_etc_indicator/ShapeShape0dense_features/Cowork_etc_indicator/Sum:output:0*
T0*
_output_shapes
:?
7dense_features/Cowork_etc_indicator/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
9dense_features/Cowork_etc_indicator/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
9dense_features/Cowork_etc_indicator/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
1dense_features/Cowork_etc_indicator/strided_sliceStridedSlice2dense_features/Cowork_etc_indicator/Shape:output:0@dense_features/Cowork_etc_indicator/strided_slice/stack:output:0Bdense_features/Cowork_etc_indicator/strided_slice/stack_1:output:0Bdense_features/Cowork_etc_indicator/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_masku
3dense_features/Cowork_etc_indicator/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
1dense_features/Cowork_etc_indicator/Reshape/shapePack:dense_features/Cowork_etc_indicator/strided_slice:output:0<dense_features/Cowork_etc_indicator/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
+dense_features/Cowork_etc_indicator/ReshapeReshape0dense_features/Cowork_etc_indicator/Sum:output:0:dense_features/Cowork_etc_indicator/Reshape/shape:output:0*
T0*'
_output_shapes
:??????????
Cdense_features/Econ_Social_indicator/to_sparse_input/ignore_value/xConst*
_output_shapes
: *
dtype0*
valueB :
??????????
Adense_features/Econ_Social_indicator/to_sparse_input/ignore_valueCastLdense_features/Econ_Social_indicator/to_sparse_input/ignore_value/x:output:0*

DstT0	*

SrcT0*
_output_shapes
: ?
=dense_features/Econ_Social_indicator/to_sparse_input/NotEqualNotEqualinputs_econ_socialEdense_features/Econ_Social_indicator/to_sparse_input/ignore_value:y:0*
T0	*'
_output_shapes
:??????????
<dense_features/Econ_Social_indicator/to_sparse_input/indicesWhereAdense_features/Econ_Social_indicator/to_sparse_input/NotEqual:z:0*'
_output_shapes
:??????????
;dense_features/Econ_Social_indicator/to_sparse_input/valuesGatherNdinputs_econ_socialDdense_features/Econ_Social_indicator/to_sparse_input/indices:index:0*
Tindices0	*
Tparams0	*#
_output_shapes
:??????????
@dense_features/Econ_Social_indicator/to_sparse_input/dense_shapeShapeinputs_econ_social*
T0	*
_output_shapes
:*
out_type0	?
Bdense_features/Econ_Social_indicator/None_Lookup/LookupTableFindV2LookupTableFindV2Odense_features_econ_social_indicator_none_lookup_lookuptablefindv2_table_handleDdense_features/Econ_Social_indicator/to_sparse_input/values:output:0Pdense_features_econ_social_indicator_none_lookup_lookuptablefindv2_default_value*	
Tin0	*

Tout0	*#
_output_shapes
:??????????
@dense_features/Econ_Social_indicator/SparseToDense/default_valueConst*
_output_shapes
: *
dtype0	*
valueB	 R
??????????
2dense_features/Econ_Social_indicator/SparseToDenseSparseToDenseDdense_features/Econ_Social_indicator/to_sparse_input/indices:index:0Idense_features/Econ_Social_indicator/to_sparse_input/dense_shape:output:0Kdense_features/Econ_Social_indicator/None_Lookup/LookupTableFindV2:values:0Idense_features/Econ_Social_indicator/SparseToDense/default_value:output:0*
T0	*
Tindices0	*'
_output_shapes
:?????????w
2dense_features/Econ_Social_indicator/one_hot/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??y
4dense_features/Econ_Social_indicator/one_hot/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    t
2dense_features/Econ_Social_indicator/one_hot/depthConst*
_output_shapes
: *
dtype0*
value	B :?
,dense_features/Econ_Social_indicator/one_hotOneHot:dense_features/Econ_Social_indicator/SparseToDense:dense:0;dense_features/Econ_Social_indicator/one_hot/depth:output:0;dense_features/Econ_Social_indicator/one_hot/Const:output:0=dense_features/Econ_Social_indicator/one_hot/Const_1:output:0*
T0*+
_output_shapes
:??????????
:dense_features/Econ_Social_indicator/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
??????????
(dense_features/Econ_Social_indicator/SumSum5dense_features/Econ_Social_indicator/one_hot:output:0Cdense_features/Econ_Social_indicator/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:??????????
*dense_features/Econ_Social_indicator/ShapeShape1dense_features/Econ_Social_indicator/Sum:output:0*
T0*
_output_shapes
:?
8dense_features/Econ_Social_indicator/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
:dense_features/Econ_Social_indicator/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
:dense_features/Econ_Social_indicator/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
2dense_features/Econ_Social_indicator/strided_sliceStridedSlice3dense_features/Econ_Social_indicator/Shape:output:0Adense_features/Econ_Social_indicator/strided_slice/stack:output:0Cdense_features/Econ_Social_indicator/strided_slice/stack_1:output:0Cdense_features/Econ_Social_indicator/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskv
4dense_features/Econ_Social_indicator/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
2dense_features/Econ_Social_indicator/Reshape/shapePack;dense_features/Econ_Social_indicator/strided_slice:output:0=dense_features/Econ_Social_indicator/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
,dense_features/Econ_Social_indicator/ReshapeReshape1dense_features/Econ_Social_indicator/Sum:output:0;dense_features/Econ_Social_indicator/Reshape/shape:output:0*
T0*'
_output_shapes
:??????????
Bdense_features/Green_Tech_indicator/to_sparse_input/ignore_value/xConst*
_output_shapes
: *
dtype0*
valueB :
??????????
@dense_features/Green_Tech_indicator/to_sparse_input/ignore_valueCastKdense_features/Green_Tech_indicator/to_sparse_input/ignore_value/x:output:0*

DstT0	*

SrcT0*
_output_shapes
: ?
<dense_features/Green_Tech_indicator/to_sparse_input/NotEqualNotEqualinputs_green_techDdense_features/Green_Tech_indicator/to_sparse_input/ignore_value:y:0*
T0	*'
_output_shapes
:??????????
;dense_features/Green_Tech_indicator/to_sparse_input/indicesWhere@dense_features/Green_Tech_indicator/to_sparse_input/NotEqual:z:0*'
_output_shapes
:??????????
:dense_features/Green_Tech_indicator/to_sparse_input/valuesGatherNdinputs_green_techCdense_features/Green_Tech_indicator/to_sparse_input/indices:index:0*
Tindices0	*
Tparams0	*#
_output_shapes
:??????????
?dense_features/Green_Tech_indicator/to_sparse_input/dense_shapeShapeinputs_green_tech*
T0	*
_output_shapes
:*
out_type0	?
Adense_features/Green_Tech_indicator/None_Lookup/LookupTableFindV2LookupTableFindV2Ndense_features_green_tech_indicator_none_lookup_lookuptablefindv2_table_handleCdense_features/Green_Tech_indicator/to_sparse_input/values:output:0Odense_features_green_tech_indicator_none_lookup_lookuptablefindv2_default_value*	
Tin0	*

Tout0	*#
_output_shapes
:??????????
?dense_features/Green_Tech_indicator/SparseToDense/default_valueConst*
_output_shapes
: *
dtype0	*
valueB	 R
??????????
1dense_features/Green_Tech_indicator/SparseToDenseSparseToDenseCdense_features/Green_Tech_indicator/to_sparse_input/indices:index:0Hdense_features/Green_Tech_indicator/to_sparse_input/dense_shape:output:0Jdense_features/Green_Tech_indicator/None_Lookup/LookupTableFindV2:values:0Hdense_features/Green_Tech_indicator/SparseToDense/default_value:output:0*
T0	*
Tindices0	*'
_output_shapes
:?????????v
1dense_features/Green_Tech_indicator/one_hot/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??x
3dense_features/Green_Tech_indicator/one_hot/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    s
1dense_features/Green_Tech_indicator/one_hot/depthConst*
_output_shapes
: *
dtype0*
value	B :<?
+dense_features/Green_Tech_indicator/one_hotOneHot9dense_features/Green_Tech_indicator/SparseToDense:dense:0:dense_features/Green_Tech_indicator/one_hot/depth:output:0:dense_features/Green_Tech_indicator/one_hot/Const:output:0<dense_features/Green_Tech_indicator/one_hot/Const_1:output:0*
T0*+
_output_shapes
:?????????<?
9dense_features/Green_Tech_indicator/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
??????????
'dense_features/Green_Tech_indicator/SumSum4dense_features/Green_Tech_indicator/one_hot:output:0Bdense_features/Green_Tech_indicator/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:?????????<?
)dense_features/Green_Tech_indicator/ShapeShape0dense_features/Green_Tech_indicator/Sum:output:0*
T0*
_output_shapes
:?
7dense_features/Green_Tech_indicator/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
9dense_features/Green_Tech_indicator/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
9dense_features/Green_Tech_indicator/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
1dense_features/Green_Tech_indicator/strided_sliceStridedSlice2dense_features/Green_Tech_indicator/Shape:output:0@dense_features/Green_Tech_indicator/strided_slice/stack:output:0Bdense_features/Green_Tech_indicator/strided_slice/stack_1:output:0Bdense_features/Green_Tech_indicator/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_masku
3dense_features/Green_Tech_indicator/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :<?
1dense_features/Green_Tech_indicator/Reshape/shapePack:dense_features/Green_Tech_indicator/strided_slice:output:0<dense_features/Green_Tech_indicator/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
+dense_features/Green_Tech_indicator/ReshapeReshape0dense_features/Green_Tech_indicator/Sum:output:0:dense_features/Green_Tech_indicator/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????<d
!dense_features/Log_Duration/ShapeShapeinputs_log_duration*
T0*
_output_shapes
:y
/dense_features/Log_Duration/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: {
1dense_features/Log_Duration/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:{
1dense_features/Log_Duration/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
)dense_features/Log_Duration/strided_sliceStridedSlice*dense_features/Log_Duration/Shape:output:08dense_features/Log_Duration/strided_slice/stack:output:0:dense_features/Log_Duration/strided_slice/stack_1:output:0:dense_features/Log_Duration/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskm
+dense_features/Log_Duration/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
)dense_features/Log_Duration/Reshape/shapePack2dense_features/Log_Duration/strided_slice:output:04dense_features/Log_Duration/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
#dense_features/Log_Duration/ReshapeReshapeinputs_log_duration2dense_features/Log_Duration/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????d
!dense_features/Log_RnD_Fund/ShapeShapeinputs_log_rnd_fund*
T0*
_output_shapes
:y
/dense_features/Log_RnD_Fund/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: {
1dense_features/Log_RnD_Fund/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:{
1dense_features/Log_RnD_Fund/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
)dense_features/Log_RnD_Fund/strided_sliceStridedSlice*dense_features/Log_RnD_Fund/Shape:output:08dense_features/Log_RnD_Fund/strided_slice/stack:output:0:dense_features/Log_RnD_Fund/strided_slice/stack_1:output:0:dense_features/Log_RnD_Fund/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskm
+dense_features/Log_RnD_Fund/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
)dense_features/Log_RnD_Fund/Reshape/shapePack2dense_features/Log_RnD_Fund/strided_slice:output:04dense_features/Log_RnD_Fund/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
#dense_features/Log_RnD_Fund/ReshapeReshapeinputs_log_rnd_fund2dense_features/Log_RnD_Fund/Reshape/shape:output:0*
T0*'
_output_shapes
:??????????
Bdense_features/Multi_Year_indicator/to_sparse_input/ignore_value/xConst*
_output_shapes
: *
dtype0*
valueB :
??????????
@dense_features/Multi_Year_indicator/to_sparse_input/ignore_valueCastKdense_features/Multi_Year_indicator/to_sparse_input/ignore_value/x:output:0*

DstT0	*

SrcT0*
_output_shapes
: ?
<dense_features/Multi_Year_indicator/to_sparse_input/NotEqualNotEqualinputs_multi_yearDdense_features/Multi_Year_indicator/to_sparse_input/ignore_value:y:0*
T0	*'
_output_shapes
:??????????
;dense_features/Multi_Year_indicator/to_sparse_input/indicesWhere@dense_features/Multi_Year_indicator/to_sparse_input/NotEqual:z:0*'
_output_shapes
:??????????
:dense_features/Multi_Year_indicator/to_sparse_input/valuesGatherNdinputs_multi_yearCdense_features/Multi_Year_indicator/to_sparse_input/indices:index:0*
Tindices0	*
Tparams0	*#
_output_shapes
:??????????
?dense_features/Multi_Year_indicator/to_sparse_input/dense_shapeShapeinputs_multi_year*
T0	*
_output_shapes
:*
out_type0	?
Adense_features/Multi_Year_indicator/None_Lookup/LookupTableFindV2LookupTableFindV2Ndense_features_multi_year_indicator_none_lookup_lookuptablefindv2_table_handleCdense_features/Multi_Year_indicator/to_sparse_input/values:output:0Odense_features_multi_year_indicator_none_lookup_lookuptablefindv2_default_value*	
Tin0	*

Tout0	*#
_output_shapes
:??????????
?dense_features/Multi_Year_indicator/SparseToDense/default_valueConst*
_output_shapes
: *
dtype0	*
valueB	 R
??????????
1dense_features/Multi_Year_indicator/SparseToDenseSparseToDenseCdense_features/Multi_Year_indicator/to_sparse_input/indices:index:0Hdense_features/Multi_Year_indicator/to_sparse_input/dense_shape:output:0Jdense_features/Multi_Year_indicator/None_Lookup/LookupTableFindV2:values:0Hdense_features/Multi_Year_indicator/SparseToDense/default_value:output:0*
T0	*
Tindices0	*'
_output_shapes
:?????????v
1dense_features/Multi_Year_indicator/one_hot/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??x
3dense_features/Multi_Year_indicator/one_hot/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    s
1dense_features/Multi_Year_indicator/one_hot/depthConst*
_output_shapes
: *
dtype0*
value	B :?
+dense_features/Multi_Year_indicator/one_hotOneHot9dense_features/Multi_Year_indicator/SparseToDense:dense:0:dense_features/Multi_Year_indicator/one_hot/depth:output:0:dense_features/Multi_Year_indicator/one_hot/Const:output:0<dense_features/Multi_Year_indicator/one_hot/Const_1:output:0*
T0*+
_output_shapes
:??????????
9dense_features/Multi_Year_indicator/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
??????????
'dense_features/Multi_Year_indicator/SumSum4dense_features/Multi_Year_indicator/one_hot:output:0Bdense_features/Multi_Year_indicator/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:??????????
)dense_features/Multi_Year_indicator/ShapeShape0dense_features/Multi_Year_indicator/Sum:output:0*
T0*
_output_shapes
:?
7dense_features/Multi_Year_indicator/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
9dense_features/Multi_Year_indicator/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
9dense_features/Multi_Year_indicator/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
1dense_features/Multi_Year_indicator/strided_sliceStridedSlice2dense_features/Multi_Year_indicator/Shape:output:0@dense_features/Multi_Year_indicator/strided_slice/stack:output:0Bdense_features/Multi_Year_indicator/strided_slice/stack_1:output:0Bdense_features/Multi_Year_indicator/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_masku
3dense_features/Multi_Year_indicator/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
1dense_features/Multi_Year_indicator/Reshape/shapePack:dense_features/Multi_Year_indicator/strided_slice:output:0<dense_features/Multi_Year_indicator/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
+dense_features/Multi_Year_indicator/ReshapeReshape0dense_features/Multi_Year_indicator/Sum:output:0:dense_features/Multi_Year_indicator/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????d
!dense_features/N_Patent_App/ShapeShapeinputs_n_patent_app*
T0*
_output_shapes
:y
/dense_features/N_Patent_App/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: {
1dense_features/N_Patent_App/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:{
1dense_features/N_Patent_App/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
)dense_features/N_Patent_App/strided_sliceStridedSlice*dense_features/N_Patent_App/Shape:output:08dense_features/N_Patent_App/strided_slice/stack:output:0:dense_features/N_Patent_App/strided_slice/stack_1:output:0:dense_features/N_Patent_App/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskm
+dense_features/N_Patent_App/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
)dense_features/N_Patent_App/Reshape/shapePack2dense_features/N_Patent_App/strided_slice:output:04dense_features/N_Patent_App/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
#dense_features/N_Patent_App/ReshapeReshapeinputs_n_patent_app2dense_features/N_Patent_App/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????d
!dense_features/N_Patent_Reg/ShapeShapeinputs_n_patent_reg*
T0*
_output_shapes
:y
/dense_features/N_Patent_Reg/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: {
1dense_features/N_Patent_Reg/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:{
1dense_features/N_Patent_Reg/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
)dense_features/N_Patent_Reg/strided_sliceStridedSlice*dense_features/N_Patent_Reg/Shape:output:08dense_features/N_Patent_Reg/strided_slice/stack:output:0:dense_features/N_Patent_Reg/strided_slice/stack_1:output:0:dense_features/N_Patent_Reg/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskm
+dense_features/N_Patent_Reg/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
)dense_features/N_Patent_Reg/Reshape/shapePack2dense_features/N_Patent_Reg/strided_slice:output:04dense_features/N_Patent_Reg/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
#dense_features/N_Patent_Reg/ReshapeReshapeinputs_n_patent_reg2dense_features/N_Patent_Reg/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????p
'dense_features/N_of_Korean_Patent/ShapeShapeinputs_n_of_korean_patent*
T0*
_output_shapes
:
5dense_features/N_of_Korean_Patent/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
7dense_features/N_of_Korean_Patent/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
7dense_features/N_of_Korean_Patent/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
/dense_features/N_of_Korean_Patent/strided_sliceStridedSlice0dense_features/N_of_Korean_Patent/Shape:output:0>dense_features/N_of_Korean_Patent/strided_slice/stack:output:0@dense_features/N_of_Korean_Patent/strided_slice/stack_1:output:0@dense_features/N_of_Korean_Patent/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_masks
1dense_features/N_of_Korean_Patent/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
/dense_features/N_of_Korean_Patent/Reshape/shapePack8dense_features/N_of_Korean_Patent/strided_slice:output:0:dense_features/N_of_Korean_Patent/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
)dense_features/N_of_Korean_Patent/ReshapeReshapeinputs_n_of_korean_patent8dense_features/N_of_Korean_Patent/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????`
dense_features/N_of_Paper/ShapeShapeinputs_n_of_paper*
T0*
_output_shapes
:w
-dense_features/N_of_Paper/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: y
/dense_features/N_of_Paper/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:y
/dense_features/N_of_Paper/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
'dense_features/N_of_Paper/strided_sliceStridedSlice(dense_features/N_of_Paper/Shape:output:06dense_features/N_of_Paper/strided_slice/stack:output:08dense_features/N_of_Paper/strided_slice/stack_1:output:08dense_features/N_of_Paper/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskk
)dense_features/N_of_Paper/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
'dense_features/N_of_Paper/Reshape/shapePack0dense_features/N_of_Paper/strided_slice:output:02dense_features/N_of_Paper/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
!dense_features/N_of_Paper/ReshapeReshapeinputs_n_of_paper0dense_features/N_of_Paper/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????\
dense_features/N_of_SCI/ShapeShapeinputs_n_of_sci*
T0*
_output_shapes
:u
+dense_features/N_of_SCI/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: w
-dense_features/N_of_SCI/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:w
-dense_features/N_of_SCI/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
%dense_features/N_of_SCI/strided_sliceStridedSlice&dense_features/N_of_SCI/Shape:output:04dense_features/N_of_SCI/strided_slice/stack:output:06dense_features/N_of_SCI/strided_slice/stack_1:output:06dense_features/N_of_SCI/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maski
'dense_features/N_of_SCI/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
%dense_features/N_of_SCI/Reshape/shapePack.dense_features/N_of_SCI/strided_slice:output:00dense_features/N_of_SCI/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
dense_features/N_of_SCI/ReshapeReshapeinputs_n_of_sci.dense_features/N_of_SCI/Reshape/shape:output:0*
T0*'
_output_shapes
:??????????
Kdense_features/National_Strategy_2_indicator/to_sparse_input/ignore_value/xConst*
_output_shapes
: *
dtype0*
valueB :
??????????
Idense_features/National_Strategy_2_indicator/to_sparse_input/ignore_valueCastTdense_features/National_Strategy_2_indicator/to_sparse_input/ignore_value/x:output:0*

DstT0	*

SrcT0*
_output_shapes
: ?
Edense_features/National_Strategy_2_indicator/to_sparse_input/NotEqualNotEqualinputs_national_strategy_2Mdense_features/National_Strategy_2_indicator/to_sparse_input/ignore_value:y:0*
T0	*'
_output_shapes
:??????????
Ddense_features/National_Strategy_2_indicator/to_sparse_input/indicesWhereIdense_features/National_Strategy_2_indicator/to_sparse_input/NotEqual:z:0*'
_output_shapes
:??????????
Cdense_features/National_Strategy_2_indicator/to_sparse_input/valuesGatherNdinputs_national_strategy_2Ldense_features/National_Strategy_2_indicator/to_sparse_input/indices:index:0*
Tindices0	*
Tparams0	*#
_output_shapes
:??????????
Hdense_features/National_Strategy_2_indicator/to_sparse_input/dense_shapeShapeinputs_national_strategy_2*
T0	*
_output_shapes
:*
out_type0	?
Jdense_features/National_Strategy_2_indicator/None_Lookup/LookupTableFindV2LookupTableFindV2Wdense_features_national_strategy_2_indicator_none_lookup_lookuptablefindv2_table_handleLdense_features/National_Strategy_2_indicator/to_sparse_input/values:output:0Xdense_features_national_strategy_2_indicator_none_lookup_lookuptablefindv2_default_value*	
Tin0	*

Tout0	*#
_output_shapes
:??????????
Hdense_features/National_Strategy_2_indicator/SparseToDense/default_valueConst*
_output_shapes
: *
dtype0	*
valueB	 R
??????????
:dense_features/National_Strategy_2_indicator/SparseToDenseSparseToDenseLdense_features/National_Strategy_2_indicator/to_sparse_input/indices:index:0Qdense_features/National_Strategy_2_indicator/to_sparse_input/dense_shape:output:0Sdense_features/National_Strategy_2_indicator/None_Lookup/LookupTableFindV2:values:0Qdense_features/National_Strategy_2_indicator/SparseToDense/default_value:output:0*
T0	*
Tindices0	*'
_output_shapes
:?????????
:dense_features/National_Strategy_2_indicator/one_hot/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
<dense_features/National_Strategy_2_indicator/one_hot/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    |
:dense_features/National_Strategy_2_indicator/one_hot/depthConst*
_output_shapes
: *
dtype0*
value	B :?
4dense_features/National_Strategy_2_indicator/one_hotOneHotBdense_features/National_Strategy_2_indicator/SparseToDense:dense:0Cdense_features/National_Strategy_2_indicator/one_hot/depth:output:0Cdense_features/National_Strategy_2_indicator/one_hot/Const:output:0Edense_features/National_Strategy_2_indicator/one_hot/Const_1:output:0*
T0*+
_output_shapes
:??????????
Bdense_features/National_Strategy_2_indicator/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
??????????
0dense_features/National_Strategy_2_indicator/SumSum=dense_features/National_Strategy_2_indicator/one_hot:output:0Kdense_features/National_Strategy_2_indicator/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:??????????
2dense_features/National_Strategy_2_indicator/ShapeShape9dense_features/National_Strategy_2_indicator/Sum:output:0*
T0*
_output_shapes
:?
@dense_features/National_Strategy_2_indicator/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
Bdense_features/National_Strategy_2_indicator/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
Bdense_features/National_Strategy_2_indicator/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
:dense_features/National_Strategy_2_indicator/strided_sliceStridedSlice;dense_features/National_Strategy_2_indicator/Shape:output:0Idense_features/National_Strategy_2_indicator/strided_slice/stack:output:0Kdense_features/National_Strategy_2_indicator/strided_slice/stack_1:output:0Kdense_features/National_Strategy_2_indicator/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask~
<dense_features/National_Strategy_2_indicator/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
:dense_features/National_Strategy_2_indicator/Reshape/shapePackCdense_features/National_Strategy_2_indicator/strided_slice:output:0Edense_features/National_Strategy_2_indicator/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
4dense_features/National_Strategy_2_indicator/ReshapeReshape9dense_features/National_Strategy_2_indicator/Sum:output:0Cdense_features/National_Strategy_2_indicator/Reshape/shape:output:0*
T0*'
_output_shapes
:??????????
?dense_features/RnD_Org_indicator/to_sparse_input/ignore_value/xConst*
_output_shapes
: *
dtype0*
valueB :
??????????
=dense_features/RnD_Org_indicator/to_sparse_input/ignore_valueCastHdense_features/RnD_Org_indicator/to_sparse_input/ignore_value/x:output:0*

DstT0	*

SrcT0*
_output_shapes
: ?
9dense_features/RnD_Org_indicator/to_sparse_input/NotEqualNotEqualinputs_rnd_orgAdense_features/RnD_Org_indicator/to_sparse_input/ignore_value:y:0*
T0	*'
_output_shapes
:??????????
8dense_features/RnD_Org_indicator/to_sparse_input/indicesWhere=dense_features/RnD_Org_indicator/to_sparse_input/NotEqual:z:0*'
_output_shapes
:??????????
7dense_features/RnD_Org_indicator/to_sparse_input/valuesGatherNdinputs_rnd_org@dense_features/RnD_Org_indicator/to_sparse_input/indices:index:0*
Tindices0	*
Tparams0	*#
_output_shapes
:??????????
<dense_features/RnD_Org_indicator/to_sparse_input/dense_shapeShapeinputs_rnd_org*
T0	*
_output_shapes
:*
out_type0	?
>dense_features/RnD_Org_indicator/None_Lookup/LookupTableFindV2LookupTableFindV2Kdense_features_rnd_org_indicator_none_lookup_lookuptablefindv2_table_handle@dense_features/RnD_Org_indicator/to_sparse_input/values:output:0Ldense_features_rnd_org_indicator_none_lookup_lookuptablefindv2_default_value*	
Tin0	*

Tout0	*#
_output_shapes
:??????????
<dense_features/RnD_Org_indicator/SparseToDense/default_valueConst*
_output_shapes
: *
dtype0	*
valueB	 R
??????????
.dense_features/RnD_Org_indicator/SparseToDenseSparseToDense@dense_features/RnD_Org_indicator/to_sparse_input/indices:index:0Edense_features/RnD_Org_indicator/to_sparse_input/dense_shape:output:0Gdense_features/RnD_Org_indicator/None_Lookup/LookupTableFindV2:values:0Edense_features/RnD_Org_indicator/SparseToDense/default_value:output:0*
T0	*
Tindices0	*'
_output_shapes
:?????????s
.dense_features/RnD_Org_indicator/one_hot/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??u
0dense_features/RnD_Org_indicator/one_hot/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    p
.dense_features/RnD_Org_indicator/one_hot/depthConst*
_output_shapes
: *
dtype0*
value	B :?
(dense_features/RnD_Org_indicator/one_hotOneHot6dense_features/RnD_Org_indicator/SparseToDense:dense:07dense_features/RnD_Org_indicator/one_hot/depth:output:07dense_features/RnD_Org_indicator/one_hot/Const:output:09dense_features/RnD_Org_indicator/one_hot/Const_1:output:0*
T0*+
_output_shapes
:??????????
6dense_features/RnD_Org_indicator/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
??????????
$dense_features/RnD_Org_indicator/SumSum1dense_features/RnD_Org_indicator/one_hot:output:0?dense_features/RnD_Org_indicator/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:??????????
&dense_features/RnD_Org_indicator/ShapeShape-dense_features/RnD_Org_indicator/Sum:output:0*
T0*
_output_shapes
:~
4dense_features/RnD_Org_indicator/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
6dense_features/RnD_Org_indicator/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
6dense_features/RnD_Org_indicator/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
.dense_features/RnD_Org_indicator/strided_sliceStridedSlice/dense_features/RnD_Org_indicator/Shape:output:0=dense_features/RnD_Org_indicator/strided_slice/stack:output:0?dense_features/RnD_Org_indicator/strided_slice/stack_1:output:0?dense_features/RnD_Org_indicator/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskr
0dense_features/RnD_Org_indicator/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
.dense_features/RnD_Org_indicator/Reshape/shapePack7dense_features/RnD_Org_indicator/strided_slice:output:09dense_features/RnD_Org_indicator/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
(dense_features/RnD_Org_indicator/ReshapeReshape-dense_features/RnD_Org_indicator/Sum:output:07dense_features/RnD_Org_indicator/Reshape/shape:output:0*
T0*'
_output_shapes
:??????????
Adense_features/RnD_Stage_indicator/to_sparse_input/ignore_value/xConst*
_output_shapes
: *
dtype0*
valueB :
??????????
?dense_features/RnD_Stage_indicator/to_sparse_input/ignore_valueCastJdense_features/RnD_Stage_indicator/to_sparse_input/ignore_value/x:output:0*

DstT0	*

SrcT0*
_output_shapes
: ?
;dense_features/RnD_Stage_indicator/to_sparse_input/NotEqualNotEqualinputs_rnd_stageCdense_features/RnD_Stage_indicator/to_sparse_input/ignore_value:y:0*
T0	*'
_output_shapes
:??????????
:dense_features/RnD_Stage_indicator/to_sparse_input/indicesWhere?dense_features/RnD_Stage_indicator/to_sparse_input/NotEqual:z:0*'
_output_shapes
:??????????
9dense_features/RnD_Stage_indicator/to_sparse_input/valuesGatherNdinputs_rnd_stageBdense_features/RnD_Stage_indicator/to_sparse_input/indices:index:0*
Tindices0	*
Tparams0	*#
_output_shapes
:??????????
>dense_features/RnD_Stage_indicator/to_sparse_input/dense_shapeShapeinputs_rnd_stage*
T0	*
_output_shapes
:*
out_type0	?
@dense_features/RnD_Stage_indicator/None_Lookup/LookupTableFindV2LookupTableFindV2Mdense_features_rnd_stage_indicator_none_lookup_lookuptablefindv2_table_handleBdense_features/RnD_Stage_indicator/to_sparse_input/values:output:0Ndense_features_rnd_stage_indicator_none_lookup_lookuptablefindv2_default_value*	
Tin0	*

Tout0	*#
_output_shapes
:??????????
>dense_features/RnD_Stage_indicator/SparseToDense/default_valueConst*
_output_shapes
: *
dtype0	*
valueB	 R
??????????
0dense_features/RnD_Stage_indicator/SparseToDenseSparseToDenseBdense_features/RnD_Stage_indicator/to_sparse_input/indices:index:0Gdense_features/RnD_Stage_indicator/to_sparse_input/dense_shape:output:0Idense_features/RnD_Stage_indicator/None_Lookup/LookupTableFindV2:values:0Gdense_features/RnD_Stage_indicator/SparseToDense/default_value:output:0*
T0	*
Tindices0	*'
_output_shapes
:?????????u
0dense_features/RnD_Stage_indicator/one_hot/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??w
2dense_features/RnD_Stage_indicator/one_hot/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    r
0dense_features/RnD_Stage_indicator/one_hot/depthConst*
_output_shapes
: *
dtype0*
value	B :?
*dense_features/RnD_Stage_indicator/one_hotOneHot8dense_features/RnD_Stage_indicator/SparseToDense:dense:09dense_features/RnD_Stage_indicator/one_hot/depth:output:09dense_features/RnD_Stage_indicator/one_hot/Const:output:0;dense_features/RnD_Stage_indicator/one_hot/Const_1:output:0*
T0*+
_output_shapes
:??????????
8dense_features/RnD_Stage_indicator/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
??????????
&dense_features/RnD_Stage_indicator/SumSum3dense_features/RnD_Stage_indicator/one_hot:output:0Adense_features/RnD_Stage_indicator/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:??????????
(dense_features/RnD_Stage_indicator/ShapeShape/dense_features/RnD_Stage_indicator/Sum:output:0*
T0*
_output_shapes
:?
6dense_features/RnD_Stage_indicator/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
8dense_features/RnD_Stage_indicator/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
8dense_features/RnD_Stage_indicator/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
0dense_features/RnD_Stage_indicator/strided_sliceStridedSlice1dense_features/RnD_Stage_indicator/Shape:output:0?dense_features/RnD_Stage_indicator/strided_slice/stack:output:0Adense_features/RnD_Stage_indicator/strided_slice/stack_1:output:0Adense_features/RnD_Stage_indicator/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskt
2dense_features/RnD_Stage_indicator/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
0dense_features/RnD_Stage_indicator/Reshape/shapePack9dense_features/RnD_Stage_indicator/strided_slice:output:0;dense_features/RnD_Stage_indicator/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
*dense_features/RnD_Stage_indicator/ReshapeReshape/dense_features/RnD_Stage_indicator/Sum:output:09dense_features/RnD_Stage_indicator/Reshape/shape:output:0*
T0*'
_output_shapes
:??????????
Cdense_features/STP_Code_11_indicator/to_sparse_input/ignore_value/xConst*
_output_shapes
: *
dtype0*
valueB B ?
=dense_features/STP_Code_11_indicator/to_sparse_input/NotEqualNotEqualinputs_stp_code_11Ldense_features/STP_Code_11_indicator/to_sparse_input/ignore_value/x:output:0*
T0*'
_output_shapes
:??????????
<dense_features/STP_Code_11_indicator/to_sparse_input/indicesWhereAdense_features/STP_Code_11_indicator/to_sparse_input/NotEqual:z:0*'
_output_shapes
:??????????
;dense_features/STP_Code_11_indicator/to_sparse_input/valuesGatherNdinputs_stp_code_11Ddense_features/STP_Code_11_indicator/to_sparse_input/indices:index:0*
Tindices0	*
Tparams0*#
_output_shapes
:??????????
@dense_features/STP_Code_11_indicator/to_sparse_input/dense_shapeShapeinputs_stp_code_11*
T0*
_output_shapes
:*
out_type0	?
Bdense_features/STP_Code_11_indicator/None_Lookup/LookupTableFindV2LookupTableFindV2Odense_features_stp_code_11_indicator_none_lookup_lookuptablefindv2_table_handleDdense_features/STP_Code_11_indicator/to_sparse_input/values:output:0Pdense_features_stp_code_11_indicator_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:??????????
@dense_features/STP_Code_11_indicator/SparseToDense/default_valueConst*
_output_shapes
: *
dtype0	*
valueB	 R
??????????
2dense_features/STP_Code_11_indicator/SparseToDenseSparseToDenseDdense_features/STP_Code_11_indicator/to_sparse_input/indices:index:0Idense_features/STP_Code_11_indicator/to_sparse_input/dense_shape:output:0Kdense_features/STP_Code_11_indicator/None_Lookup/LookupTableFindV2:values:0Idense_features/STP_Code_11_indicator/SparseToDense/default_value:output:0*
T0	*
Tindices0	*'
_output_shapes
:?????????w
2dense_features/STP_Code_11_indicator/one_hot/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??y
4dense_features/STP_Code_11_indicator/one_hot/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    u
2dense_features/STP_Code_11_indicator/one_hot/depthConst*
_output_shapes
: *
dtype0*
value
B :??
,dense_features/STP_Code_11_indicator/one_hotOneHot:dense_features/STP_Code_11_indicator/SparseToDense:dense:0;dense_features/STP_Code_11_indicator/one_hot/depth:output:0;dense_features/STP_Code_11_indicator/one_hot/Const:output:0=dense_features/STP_Code_11_indicator/one_hot/Const_1:output:0*
T0*,
_output_shapes
:???????????
:dense_features/STP_Code_11_indicator/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
??????????
(dense_features/STP_Code_11_indicator/SumSum5dense_features/STP_Code_11_indicator/one_hot:output:0Cdense_features/STP_Code_11_indicator/Sum/reduction_indices:output:0*
T0*(
_output_shapes
:???????????
*dense_features/STP_Code_11_indicator/ShapeShape1dense_features/STP_Code_11_indicator/Sum:output:0*
T0*
_output_shapes
:?
8dense_features/STP_Code_11_indicator/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
:dense_features/STP_Code_11_indicator/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
:dense_features/STP_Code_11_indicator/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
2dense_features/STP_Code_11_indicator/strided_sliceStridedSlice3dense_features/STP_Code_11_indicator/Shape:output:0Adense_features/STP_Code_11_indicator/strided_slice/stack:output:0Cdense_features/STP_Code_11_indicator/strided_slice/stack_1:output:0Cdense_features/STP_Code_11_indicator/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskw
4dense_features/STP_Code_11_indicator/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value
B :??
2dense_features/STP_Code_11_indicator/Reshape/shapePack;dense_features/STP_Code_11_indicator/strided_slice:output:0=dense_features/STP_Code_11_indicator/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
,dense_features/STP_Code_11_indicator/ReshapeReshape1dense_features/STP_Code_11_indicator/Sum:output:0;dense_features/STP_Code_11_indicator/Reshape/shape:output:0*
T0*(
_output_shapes
:??????????n
&dense_features/STP_Code_1_Weight/ShapeShapeinputs_stp_code_1_weight*
T0*
_output_shapes
:~
4dense_features/STP_Code_1_Weight/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
6dense_features/STP_Code_1_Weight/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
6dense_features/STP_Code_1_Weight/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
.dense_features/STP_Code_1_Weight/strided_sliceStridedSlice/dense_features/STP_Code_1_Weight/Shape:output:0=dense_features/STP_Code_1_Weight/strided_slice/stack:output:0?dense_features/STP_Code_1_Weight/strided_slice/stack_1:output:0?dense_features/STP_Code_1_Weight/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskr
0dense_features/STP_Code_1_Weight/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
.dense_features/STP_Code_1_Weight/Reshape/shapePack7dense_features/STP_Code_1_Weight/strided_slice:output:09dense_features/STP_Code_1_Weight/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
(dense_features/STP_Code_1_Weight/ReshapeReshapeinputs_stp_code_1_weight7dense_features/STP_Code_1_Weight/Reshape/shape:output:0*
T0*'
_output_shapes
:??????????
Cdense_features/STP_Code_21_indicator/to_sparse_input/ignore_value/xConst*
_output_shapes
: *
dtype0*
valueB B ?
=dense_features/STP_Code_21_indicator/to_sparse_input/NotEqualNotEqualinputs_stp_code_21Ldense_features/STP_Code_21_indicator/to_sparse_input/ignore_value/x:output:0*
T0*'
_output_shapes
:??????????
<dense_features/STP_Code_21_indicator/to_sparse_input/indicesWhereAdense_features/STP_Code_21_indicator/to_sparse_input/NotEqual:z:0*'
_output_shapes
:??????????
;dense_features/STP_Code_21_indicator/to_sparse_input/valuesGatherNdinputs_stp_code_21Ddense_features/STP_Code_21_indicator/to_sparse_input/indices:index:0*
Tindices0	*
Tparams0*#
_output_shapes
:??????????
@dense_features/STP_Code_21_indicator/to_sparse_input/dense_shapeShapeinputs_stp_code_21*
T0*
_output_shapes
:*
out_type0	?
Bdense_features/STP_Code_21_indicator/None_Lookup/LookupTableFindV2LookupTableFindV2Odense_features_stp_code_21_indicator_none_lookup_lookuptablefindv2_table_handleDdense_features/STP_Code_21_indicator/to_sparse_input/values:output:0Pdense_features_stp_code_21_indicator_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:??????????
@dense_features/STP_Code_21_indicator/SparseToDense/default_valueConst*
_output_shapes
: *
dtype0	*
valueB	 R
??????????
2dense_features/STP_Code_21_indicator/SparseToDenseSparseToDenseDdense_features/STP_Code_21_indicator/to_sparse_input/indices:index:0Idense_features/STP_Code_21_indicator/to_sparse_input/dense_shape:output:0Kdense_features/STP_Code_21_indicator/None_Lookup/LookupTableFindV2:values:0Idense_features/STP_Code_21_indicator/SparseToDense/default_value:output:0*
T0	*
Tindices0	*'
_output_shapes
:?????????w
2dense_features/STP_Code_21_indicator/one_hot/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??y
4dense_features/STP_Code_21_indicator/one_hot/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    u
2dense_features/STP_Code_21_indicator/one_hot/depthConst*
_output_shapes
: *
dtype0*
value
B :??
,dense_features/STP_Code_21_indicator/one_hotOneHot:dense_features/STP_Code_21_indicator/SparseToDense:dense:0;dense_features/STP_Code_21_indicator/one_hot/depth:output:0;dense_features/STP_Code_21_indicator/one_hot/Const:output:0=dense_features/STP_Code_21_indicator/one_hot/Const_1:output:0*
T0*,
_output_shapes
:???????????
:dense_features/STP_Code_21_indicator/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
??????????
(dense_features/STP_Code_21_indicator/SumSum5dense_features/STP_Code_21_indicator/one_hot:output:0Cdense_features/STP_Code_21_indicator/Sum/reduction_indices:output:0*
T0*(
_output_shapes
:???????????
*dense_features/STP_Code_21_indicator/ShapeShape1dense_features/STP_Code_21_indicator/Sum:output:0*
T0*
_output_shapes
:?
8dense_features/STP_Code_21_indicator/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
:dense_features/STP_Code_21_indicator/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
:dense_features/STP_Code_21_indicator/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
2dense_features/STP_Code_21_indicator/strided_sliceStridedSlice3dense_features/STP_Code_21_indicator/Shape:output:0Adense_features/STP_Code_21_indicator/strided_slice/stack:output:0Cdense_features/STP_Code_21_indicator/strided_slice/stack_1:output:0Cdense_features/STP_Code_21_indicator/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskw
4dense_features/STP_Code_21_indicator/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value
B :??
2dense_features/STP_Code_21_indicator/Reshape/shapePack;dense_features/STP_Code_21_indicator/strided_slice:output:0=dense_features/STP_Code_21_indicator/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
,dense_features/STP_Code_21_indicator/ReshapeReshape1dense_features/STP_Code_21_indicator/Sum:output:0;dense_features/STP_Code_21_indicator/Reshape/shape:output:0*
T0*(
_output_shapes
:??????????n
&dense_features/STP_Code_2_Weight/ShapeShapeinputs_stp_code_2_weight*
T0*
_output_shapes
:~
4dense_features/STP_Code_2_Weight/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
6dense_features/STP_Code_2_Weight/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
6dense_features/STP_Code_2_Weight/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
.dense_features/STP_Code_2_Weight/strided_sliceStridedSlice/dense_features/STP_Code_2_Weight/Shape:output:0=dense_features/STP_Code_2_Weight/strided_slice/stack:output:0?dense_features/STP_Code_2_Weight/strided_slice/stack_1:output:0?dense_features/STP_Code_2_Weight/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskr
0dense_features/STP_Code_2_Weight/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
.dense_features/STP_Code_2_Weight/Reshape/shapePack7dense_features/STP_Code_2_Weight/strided_slice:output:09dense_features/STP_Code_2_Weight/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
(dense_features/STP_Code_2_Weight/ReshapeReshapeinputs_stp_code_2_weight7dense_features/STP_Code_2_Weight/Reshape/shape:output:0*
T0*'
_output_shapes
:??????????
>dense_features/SixT_2_indicator/to_sparse_input/ignore_value/xConst*
_output_shapes
: *
dtype0*
valueB :
??????????
<dense_features/SixT_2_indicator/to_sparse_input/ignore_valueCastGdense_features/SixT_2_indicator/to_sparse_input/ignore_value/x:output:0*

DstT0	*

SrcT0*
_output_shapes
: ?
8dense_features/SixT_2_indicator/to_sparse_input/NotEqualNotEqualinputs_sixt_2@dense_features/SixT_2_indicator/to_sparse_input/ignore_value:y:0*
T0	*'
_output_shapes
:??????????
7dense_features/SixT_2_indicator/to_sparse_input/indicesWhere<dense_features/SixT_2_indicator/to_sparse_input/NotEqual:z:0*'
_output_shapes
:??????????
6dense_features/SixT_2_indicator/to_sparse_input/valuesGatherNdinputs_sixt_2?dense_features/SixT_2_indicator/to_sparse_input/indices:index:0*
Tindices0	*
Tparams0	*#
_output_shapes
:??????????
;dense_features/SixT_2_indicator/to_sparse_input/dense_shapeShapeinputs_sixt_2*
T0	*
_output_shapes
:*
out_type0	?
=dense_features/SixT_2_indicator/None_Lookup/LookupTableFindV2LookupTableFindV2Jdense_features_sixt_2_indicator_none_lookup_lookuptablefindv2_table_handle?dense_features/SixT_2_indicator/to_sparse_input/values:output:0Kdense_features_sixt_2_indicator_none_lookup_lookuptablefindv2_default_value*	
Tin0	*

Tout0	*#
_output_shapes
:??????????
;dense_features/SixT_2_indicator/SparseToDense/default_valueConst*
_output_shapes
: *
dtype0	*
valueB	 R
??????????
-dense_features/SixT_2_indicator/SparseToDenseSparseToDense?dense_features/SixT_2_indicator/to_sparse_input/indices:index:0Ddense_features/SixT_2_indicator/to_sparse_input/dense_shape:output:0Fdense_features/SixT_2_indicator/None_Lookup/LookupTableFindV2:values:0Ddense_features/SixT_2_indicator/SparseToDense/default_value:output:0*
T0	*
Tindices0	*'
_output_shapes
:?????????r
-dense_features/SixT_2_indicator/one_hot/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??t
/dense_features/SixT_2_indicator/one_hot/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    o
-dense_features/SixT_2_indicator/one_hot/depthConst*
_output_shapes
: *
dtype0*
value	B :?
'dense_features/SixT_2_indicator/one_hotOneHot5dense_features/SixT_2_indicator/SparseToDense:dense:06dense_features/SixT_2_indicator/one_hot/depth:output:06dense_features/SixT_2_indicator/one_hot/Const:output:08dense_features/SixT_2_indicator/one_hot/Const_1:output:0*
T0*+
_output_shapes
:??????????
5dense_features/SixT_2_indicator/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
??????????
#dense_features/SixT_2_indicator/SumSum0dense_features/SixT_2_indicator/one_hot:output:0>dense_features/SixT_2_indicator/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:??????????
%dense_features/SixT_2_indicator/ShapeShape,dense_features/SixT_2_indicator/Sum:output:0*
T0*
_output_shapes
:}
3dense_features/SixT_2_indicator/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
5dense_features/SixT_2_indicator/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
5dense_features/SixT_2_indicator/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
-dense_features/SixT_2_indicator/strided_sliceStridedSlice.dense_features/SixT_2_indicator/Shape:output:0<dense_features/SixT_2_indicator/strided_slice/stack:output:0>dense_features/SixT_2_indicator/strided_slice/stack_1:output:0>dense_features/SixT_2_indicator/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskq
/dense_features/SixT_2_indicator/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
-dense_features/SixT_2_indicator/Reshape/shapePack6dense_features/SixT_2_indicator/strided_slice:output:08dense_features/SixT_2_indicator/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
'dense_features/SixT_2_indicator/ReshapeReshape,dense_features/SixT_2_indicator/Sum:output:06dense_features/SixT_2_indicator/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????T
dense_features/Year/ShapeShapeinputs_year*
T0*
_output_shapes
:q
'dense_features/Year/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: s
)dense_features/Year/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:s
)dense_features/Year/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
!dense_features/Year/strided_sliceStridedSlice"dense_features/Year/Shape:output:00dense_features/Year/strided_slice/stack:output:02dense_features/Year/strided_slice/stack_1:output:02dense_features/Year/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maske
#dense_features/Year/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
!dense_features/Year/Reshape/shapePack*dense_features/Year/strided_slice:output:0,dense_features/Year/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
dense_features/Year/ReshapeReshapeinputs_year*dense_features/Year/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????e
dense_features/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
??????????
dense_features/concatConcatV29dense_features/Application_Area_1_Weight/Reshape:output:0<dense_features/Application_Area_1_indicator/Reshape:output:09dense_features/Application_Area_2_Weight/Reshape:output:0<dense_features/Application_Area_2_indicator/Reshape:output:07dense_features/Cowork_Abroad_indicator/Reshape:output:04dense_features/Cowork_Cor_indicator/Reshape:output:05dense_features/Cowork_Inst_indicator/Reshape:output:04dense_features/Cowork_Uni_indicator/Reshape:output:04dense_features/Cowork_etc_indicator/Reshape:output:05dense_features/Econ_Social_indicator/Reshape:output:04dense_features/Green_Tech_indicator/Reshape:output:0,dense_features/Log_Duration/Reshape:output:0,dense_features/Log_RnD_Fund/Reshape:output:04dense_features/Multi_Year_indicator/Reshape:output:0,dense_features/N_Patent_App/Reshape:output:0,dense_features/N_Patent_Reg/Reshape:output:02dense_features/N_of_Korean_Patent/Reshape:output:0*dense_features/N_of_Paper/Reshape:output:0(dense_features/N_of_SCI/Reshape:output:0=dense_features/National_Strategy_2_indicator/Reshape:output:01dense_features/RnD_Org_indicator/Reshape:output:03dense_features/RnD_Stage_indicator/Reshape:output:05dense_features/STP_Code_11_indicator/Reshape:output:01dense_features/STP_Code_1_Weight/Reshape:output:05dense_features/STP_Code_21_indicator/Reshape:output:01dense_features/STP_Code_2_Weight/Reshape:output:00dense_features/SixT_2_indicator/Reshape:output:0$dense_features/Year/Reshape:output:0#dense_features/concat/axis:output:0*
N*
T0*(
_output_shapes
:???????????
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype0?
dense/MatMulMatMuldense_features/concat:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????]

dense/ReluReludense/BiasAdd:output:0*
T0*(
_output_shapes
:??????????Z
dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
dropout/dropout/MulMuldense/Relu:activations:0dropout/dropout/Const:output:0*
T0*(
_output_shapes
:??????????]
dropout/dropout/ShapeShapedense/Relu:activations:0*
T0*
_output_shapes
:?
,dropout/dropout/random_uniform/RandomUniformRandomUniformdropout/dropout/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0c
dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>?
dropout/dropout/GreaterEqualGreaterEqual5dropout/dropout/random_uniform/RandomUniform:output:0'dropout/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:???????????
dropout/dropout/CastCast dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:???????????
dropout/dropout/Mul_1Muldropout/dropout/Mul:z:0dropout/dropout/Cast:y:0*
T0*(
_output_shapes
:???????????
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype0?
dense_1/MatMulMatMuldropout/dropout/Mul_1:z:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????a
dense_1/ReluReludense_1/BiasAdd:output:0*
T0*(
_output_shapes
:??????????\
dropout_1/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
dropout_1/dropout/MulMuldense_1/Relu:activations:0 dropout_1/dropout/Const:output:0*
T0*(
_output_shapes
:??????????a
dropout_1/dropout/ShapeShapedense_1/Relu:activations:0*
T0*
_output_shapes
:?
.dropout_1/dropout/random_uniform/RandomUniformRandomUniform dropout_1/dropout/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0e
 dropout_1/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>?
dropout_1/dropout/GreaterEqualGreaterEqual7dropout_1/dropout/random_uniform/RandomUniform:output:0)dropout_1/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:???????????
dropout_1/dropout/CastCast"dropout_1/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:???????????
dropout_1/dropout/Mul_1Muldropout_1/dropout/Mul:z:0dropout_1/dropout/Cast:y:0*
T0*(
_output_shapes
:???????????
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype0?
dense_2/MatMulMatMuldropout_1/dropout/Mul_1:z:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????a
dense_2/ReluReludense_2/BiasAdd:output:0*
T0*(
_output_shapes
:??????????\
dropout_2/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
dropout_2/dropout/MulMuldense_2/Relu:activations:0 dropout_2/dropout/Const:output:0*
T0*(
_output_shapes
:??????????a
dropout_2/dropout/ShapeShapedense_2/Relu:activations:0*
T0*
_output_shapes
:?
.dropout_2/dropout/random_uniform/RandomUniformRandomUniform dropout_2/dropout/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0e
 dropout_2/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>?
dropout_2/dropout/GreaterEqualGreaterEqual7dropout_2/dropout/random_uniform/RandomUniform:output:0)dropout_2/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:???????????
dropout_2/dropout/CastCast"dropout_2/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:???????????
dropout_2/dropout/Mul_1Muldropout_2/dropout/Mul:z:0dropout_2/dropout/Cast:y:0*
T0*(
_output_shapes
:???????????
dense_3/MatMul/ReadVariableOpReadVariableOp&dense_3_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype0?
dense_3/MatMulMatMuldropout_2/dropout/Mul_1:z:0%dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
dense_3/BiasAdd/ReadVariableOpReadVariableOp'dense_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
dense_3/BiasAddBiasAdddense_3/MatMul:product:0&dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????f
dense_3/SigmoidSigmoiddense_3/BiasAdd:output:0*
T0*'
_output_shapes
:?????????b
IdentityIdentitydense_3/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp^dense_3/BiasAdd/ReadVariableOp^dense_3/MatMul/ReadVariableOpJ^dense_features/Application_Area_1_indicator/None_Lookup/LookupTableFindV2J^dense_features/Application_Area_2_indicator/None_Lookup/LookupTableFindV2E^dense_features/Cowork_Abroad_indicator/None_Lookup/LookupTableFindV2B^dense_features/Cowork_Cor_indicator/None_Lookup/LookupTableFindV2C^dense_features/Cowork_Inst_indicator/None_Lookup/LookupTableFindV2B^dense_features/Cowork_Uni_indicator/None_Lookup/LookupTableFindV2B^dense_features/Cowork_etc_indicator/None_Lookup/LookupTableFindV2C^dense_features/Econ_Social_indicator/None_Lookup/LookupTableFindV2B^dense_features/Green_Tech_indicator/None_Lookup/LookupTableFindV2B^dense_features/Multi_Year_indicator/None_Lookup/LookupTableFindV2K^dense_features/National_Strategy_2_indicator/None_Lookup/LookupTableFindV2?^dense_features/RnD_Org_indicator/None_Lookup/LookupTableFindV2A^dense_features/RnD_Stage_indicator/None_Lookup/LookupTableFindV2C^dense_features/STP_Code_11_indicator/None_Lookup/LookupTableFindV2C^dense_features/STP_Code_21_indicator/None_Lookup/LookupTableFindV2>^dense_features/SixT_2_indicator/None_Lookup/LookupTableFindV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2>
dense_2/MatMul/ReadVariableOpdense_2/MatMul/ReadVariableOp2@
dense_3/BiasAdd/ReadVariableOpdense_3/BiasAdd/ReadVariableOp2>
dense_3/MatMul/ReadVariableOpdense_3/MatMul/ReadVariableOp2?
Idense_features/Application_Area_1_indicator/None_Lookup/LookupTableFindV2Idense_features/Application_Area_1_indicator/None_Lookup/LookupTableFindV22?
Idense_features/Application_Area_2_indicator/None_Lookup/LookupTableFindV2Idense_features/Application_Area_2_indicator/None_Lookup/LookupTableFindV22?
Ddense_features/Cowork_Abroad_indicator/None_Lookup/LookupTableFindV2Ddense_features/Cowork_Abroad_indicator/None_Lookup/LookupTableFindV22?
Adense_features/Cowork_Cor_indicator/None_Lookup/LookupTableFindV2Adense_features/Cowork_Cor_indicator/None_Lookup/LookupTableFindV22?
Bdense_features/Cowork_Inst_indicator/None_Lookup/LookupTableFindV2Bdense_features/Cowork_Inst_indicator/None_Lookup/LookupTableFindV22?
Adense_features/Cowork_Uni_indicator/None_Lookup/LookupTableFindV2Adense_features/Cowork_Uni_indicator/None_Lookup/LookupTableFindV22?
Adense_features/Cowork_etc_indicator/None_Lookup/LookupTableFindV2Adense_features/Cowork_etc_indicator/None_Lookup/LookupTableFindV22?
Bdense_features/Econ_Social_indicator/None_Lookup/LookupTableFindV2Bdense_features/Econ_Social_indicator/None_Lookup/LookupTableFindV22?
Adense_features/Green_Tech_indicator/None_Lookup/LookupTableFindV2Adense_features/Green_Tech_indicator/None_Lookup/LookupTableFindV22?
Adense_features/Multi_Year_indicator/None_Lookup/LookupTableFindV2Adense_features/Multi_Year_indicator/None_Lookup/LookupTableFindV22?
Jdense_features/National_Strategy_2_indicator/None_Lookup/LookupTableFindV2Jdense_features/National_Strategy_2_indicator/None_Lookup/LookupTableFindV22?
>dense_features/RnD_Org_indicator/None_Lookup/LookupTableFindV2>dense_features/RnD_Org_indicator/None_Lookup/LookupTableFindV22?
@dense_features/RnD_Stage_indicator/None_Lookup/LookupTableFindV2@dense_features/RnD_Stage_indicator/None_Lookup/LookupTableFindV22?
Bdense_features/STP_Code_11_indicator/None_Lookup/LookupTableFindV2Bdense_features/STP_Code_11_indicator/None_Lookup/LookupTableFindV22?
Bdense_features/STP_Code_21_indicator/None_Lookup/LookupTableFindV2Bdense_features/STP_Code_21_indicator/None_Lookup/LookupTableFindV22~
=dense_features/SixT_2_indicator/None_Lookup/LookupTableFindV2=dense_features/SixT_2_indicator/None_Lookup/LookupTableFindV2:b ^
'
_output_shapes
:?????????
3
_user_specified_nameinputs/Application_Area_1:ie
'
_output_shapes
:?????????
:
_user_specified_name" inputs/Application_Area_1_Weight:b^
'
_output_shapes
:?????????
3
_user_specified_nameinputs/Application_Area_2:ie
'
_output_shapes
:?????????
:
_user_specified_name" inputs/Application_Area_2_Weight:]Y
'
_output_shapes
:?????????
.
_user_specified_nameinputs/Cowork_Abroad:ZV
'
_output_shapes
:?????????
+
_user_specified_nameinputs/Cowork_Cor:[W
'
_output_shapes
:?????????
,
_user_specified_nameinputs/Cowork_Inst:ZV
'
_output_shapes
:?????????
+
_user_specified_nameinputs/Cowork_Uni:ZV
'
_output_shapes
:?????????
+
_user_specified_nameinputs/Cowork_etc:[	W
'
_output_shapes
:?????????
,
_user_specified_nameinputs/Econ_Social:Z
V
'
_output_shapes
:?????????
+
_user_specified_nameinputs/Green_Tech:\X
'
_output_shapes
:?????????
-
_user_specified_nameinputs/Log_Duration:\X
'
_output_shapes
:?????????
-
_user_specified_nameinputs/Log_RnD_Fund:ZV
'
_output_shapes
:?????????
+
_user_specified_nameinputs/Multi_Year:\X
'
_output_shapes
:?????????
-
_user_specified_nameinputs/N_Patent_App:\X
'
_output_shapes
:?????????
-
_user_specified_nameinputs/N_Patent_Reg:b^
'
_output_shapes
:?????????
3
_user_specified_nameinputs/N_of_Korean_Patent:ZV
'
_output_shapes
:?????????
+
_user_specified_nameinputs/N_of_Paper:XT
'
_output_shapes
:?????????
)
_user_specified_nameinputs/N_of_SCI:c_
'
_output_shapes
:?????????
4
_user_specified_nameinputs/National_Strategy_2:WS
'
_output_shapes
:?????????
(
_user_specified_nameinputs/RnD_Org:YU
'
_output_shapes
:?????????
*
_user_specified_nameinputs/RnD_Stage:[W
'
_output_shapes
:?????????
,
_user_specified_nameinputs/STP_Code_11:a]
'
_output_shapes
:?????????
2
_user_specified_nameinputs/STP_Code_1_Weight:[W
'
_output_shapes
:?????????
,
_user_specified_nameinputs/STP_Code_21:a]
'
_output_shapes
:?????????
2
_user_specified_nameinputs/STP_Code_2_Weight:VR
'
_output_shapes
:?????????
'
_user_specified_nameinputs/SixT_2:TP
'
_output_shapes
:?????????
%
_user_specified_nameinputs/Year:

_output_shapes
: :

_output_shapes
: :!

_output_shapes
: :#

_output_shapes
: :%

_output_shapes
: :'

_output_shapes
: :)

_output_shapes
: :+

_output_shapes
: :-

_output_shapes
: :/

_output_shapes
: :1

_output_shapes
: :3

_output_shapes
: :5

_output_shapes
: :7

_output_shapes
: :9

_output_shapes
: :;

_output_shapes
: 
?
-
__inference__destroyer_133704
identityG
ConstConst*
_output_shapes
: *
dtype0*
value	B :E
IdentityIdentityConst:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 
?
-
__inference__destroyer_133722
identityG
ConstConst*
_output_shapes
: *
dtype0*
value	B :E
IdentityIdentityConst:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 
?
?
__inference__initializer_1337532
.table_init477_lookuptableimportv2_table_handle*
&table_init477_lookuptableimportv2_keys,
(table_init477_lookuptableimportv2_values	
identity??!table_init477/LookupTableImportV2?
!table_init477/LookupTableImportV2LookupTableImportV2.table_init477_lookuptableimportv2_table_handle&table_init477_lookuptableimportv2_keys(table_init477_lookuptableimportv2_values*	
Tin0*

Tout0	*
_output_shapes
 G
ConstConst*
_output_shapes
: *
dtype0*
value	B :L
IdentityIdentityConst:output:0^NoOp*
T0*
_output_shapes
: j
NoOpNoOp"^table_init477/LookupTableImportV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*!
_input_shapes
: ::2F
!table_init477/LookupTableImportV2!table_init477/LookupTableImportV2: 

_output_shapes
:: 

_output_shapes
:
?
-
__inference__destroyer_133866
identityG
ConstConst*
_output_shapes
: *
dtype0*
value	B :E
IdentityIdentityConst:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 
?
;
__inference__creator_133799
identity??
hash_tablek

hash_tableHashTableV2*
_output_shapes
: *
	key_dtype0	*
shared_name584*
value_dtype0	W
IdentityIdentityhash_table:table_handle:0^NoOp*
T0*
_output_shapes
: S
NoOpNoOp^hash_table*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 2

hash_table
hash_table
?
?
__inference__initializer_1338252
.table_init635_lookuptableimportv2_table_handle*
&table_init635_lookuptableimportv2_keys	,
(table_init635_lookuptableimportv2_values	
identity??!table_init635/LookupTableImportV2?
!table_init635/LookupTableImportV2LookupTableImportV2.table_init635_lookuptableimportv2_table_handle&table_init635_lookuptableimportv2_keys(table_init635_lookuptableimportv2_values*	
Tin0	*

Tout0	*
_output_shapes
 G
ConstConst*
_output_shapes
: *
dtype0*
value	B :L
IdentityIdentityConst:output:0^NoOp*
T0*
_output_shapes
: j
NoOpNoOp"^table_init635/LookupTableImportV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*!
_input_shapes
: ::2F
!table_init635/LookupTableImportV2!table_init635/LookupTableImportV2: 

_output_shapes
:: 

_output_shapes
:
?
?
__inference__initializer_1339332
.table_init903_lookuptableimportv2_table_handle*
&table_init903_lookuptableimportv2_keys	,
(table_init903_lookuptableimportv2_values	
identity??!table_init903/LookupTableImportV2?
!table_init903/LookupTableImportV2LookupTableImportV2.table_init903_lookuptableimportv2_table_handle&table_init903_lookuptableimportv2_keys(table_init903_lookuptableimportv2_values*	
Tin0	*

Tout0	*
_output_shapes
 G
ConstConst*
_output_shapes
: *
dtype0*
value	B :L
IdentityIdentityConst:output:0^NoOp*
T0*
_output_shapes
: j
NoOpNoOp"^table_init903/LookupTableImportV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*!
_input_shapes
: ::2F
!table_init903/LookupTableImportV2!table_init903/LookupTableImportV2: 

_output_shapes
:: 

_output_shapes
:
?
;
__inference__creator_133655
identity??
hash_tablek

hash_tableHashTableV2*
_output_shapes
: *
	key_dtype0*
shared_name300*
value_dtype0	W
IdentityIdentityhash_table:table_handle:0^NoOp*
T0*
_output_shapes
: S
NoOpNoOp^hash_table*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 2

hash_table
hash_table
?
c
E__inference_dropout_1_layer_call_and_return_conditional_losses_129148

inputs

identity_1O
IdentityIdentityinputs*
T0*(
_output_shapes
:??????????\

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:??????????"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?Q
?
F__inference_sequential_layer_call_and_return_conditional_losses_130671
application_area_1
application_area_1_weight
application_area_2
application_area_2_weight
cowork_abroad

cowork_cor
cowork_inst

cowork_uni

cowork_etc
econ_social	

green_tech	
log_duration
log_rnd_fund

multi_year	
n_patent_app
n_patent_reg
n_of_korean_patent

n_of_paper
n_of_sci
national_strategy_2	
rnd_org	
	rnd_stage	
stp_code_11
stp_code_1_weight
stp_code_21
stp_code_2_weight

sixt_2	
year
dense_features_130582
dense_features_130584	
dense_features_130586
dense_features_130588	
dense_features_130590
dense_features_130592	
dense_features_130594
dense_features_130596	
dense_features_130598
dense_features_130600	
dense_features_130602
dense_features_130604	
dense_features_130606
dense_features_130608	
dense_features_130610
dense_features_130612	
dense_features_130614
dense_features_130616	
dense_features_130618
dense_features_130620	
dense_features_130622
dense_features_130624	
dense_features_130626
dense_features_130628	
dense_features_130630
dense_features_130632	
dense_features_130634
dense_features_130636	
dense_features_130638
dense_features_130640	
dense_features_130642
dense_features_130644	 
dense_130647:
??
dense_130649:	?"
dense_1_130653:
??
dense_1_130655:	?"
dense_2_130659:
??
dense_2_130661:	?!
dense_3_130665:	?
dense_3_130667:
identity??dense/StatefulPartitionedCall?dense_1/StatefulPartitionedCall?dense_2/StatefulPartitionedCall?dense_3/StatefulPartitionedCall?&dense_features/StatefulPartitionedCall?
&dense_features/StatefulPartitionedCallStatefulPartitionedCallapplication_area_1application_area_1_weightapplication_area_2application_area_2_weightcowork_abroad
cowork_corcowork_inst
cowork_uni
cowork_etcecon_social
green_techlog_durationlog_rnd_fund
multi_yearn_patent_appn_patent_regn_of_korean_patent
n_of_papern_of_scinational_strategy_2rnd_org	rnd_stagestp_code_11stp_code_1_weightstp_code_21stp_code_2_weightsixt_2yeardense_features_130582dense_features_130584dense_features_130586dense_features_130588dense_features_130590dense_features_130592dense_features_130594dense_features_130596dense_features_130598dense_features_130600dense_features_130602dense_features_130604dense_features_130606dense_features_130608dense_features_130610dense_features_130612dense_features_130614dense_features_130616dense_features_130618dense_features_130620dense_features_130622dense_features_130624dense_features_130626dense_features_130628dense_features_130630dense_features_130632dense_features_130634dense_features_130636dense_features_130638dense_features_130640dense_features_130642dense_features_130644*G
Tin@
>2<																							*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_dense_features_layer_call_and_return_conditional_losses_129036?
dense/StatefulPartitionedCallStatefulPartitionedCall/dense_features/StatefulPartitionedCall:output:0dense_130647dense_130649*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_129113?
dropout/PartitionedCallPartitionedCall&dense/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_dropout_layer_call_and_return_conditional_losses_129124?
dense_1/StatefulPartitionedCallStatefulPartitionedCall dropout/PartitionedCall:output:0dense_1_130653dense_1_130655*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_129137?
dropout_1/PartitionedCallPartitionedCall(dense_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_dropout_1_layer_call_and_return_conditional_losses_129148?
dense_2/StatefulPartitionedCallStatefulPartitionedCall"dropout_1/PartitionedCall:output:0dense_2_130659dense_2_130661*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_dense_2_layer_call_and_return_conditional_losses_129161?
dropout_2/PartitionedCallPartitionedCall(dense_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_dropout_2_layer_call_and_return_conditional_losses_129172?
dense_3/StatefulPartitionedCallStatefulPartitionedCall"dropout_2/PartitionedCall:output:0dense_3_130665dense_3_130667*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_dense_3_layer_call_and_return_conditional_losses_129185w
IdentityIdentity(dense_3/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall'^dense_features/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall2P
&dense_features/StatefulPartitionedCall&dense_features/StatefulPartitionedCall:[ W
'
_output_shapes
:?????????
,
_user_specified_nameApplication_Area_1:b^
'
_output_shapes
:?????????
3
_user_specified_nameApplication_Area_1_Weight:[W
'
_output_shapes
:?????????
,
_user_specified_nameApplication_Area_2:b^
'
_output_shapes
:?????????
3
_user_specified_nameApplication_Area_2_Weight:VR
'
_output_shapes
:?????????
'
_user_specified_nameCowork_Abroad:SO
'
_output_shapes
:?????????
$
_user_specified_name
Cowork_Cor:TP
'
_output_shapes
:?????????
%
_user_specified_nameCowork_Inst:SO
'
_output_shapes
:?????????
$
_user_specified_name
Cowork_Uni:SO
'
_output_shapes
:?????????
$
_user_specified_name
Cowork_etc:T	P
'
_output_shapes
:?????????
%
_user_specified_nameEcon_Social:S
O
'
_output_shapes
:?????????
$
_user_specified_name
Green_Tech:UQ
'
_output_shapes
:?????????
&
_user_specified_nameLog_Duration:UQ
'
_output_shapes
:?????????
&
_user_specified_nameLog_RnD_Fund:SO
'
_output_shapes
:?????????
$
_user_specified_name
Multi_Year:UQ
'
_output_shapes
:?????????
&
_user_specified_nameN_Patent_App:UQ
'
_output_shapes
:?????????
&
_user_specified_nameN_Patent_Reg:[W
'
_output_shapes
:?????????
,
_user_specified_nameN_of_Korean_Patent:SO
'
_output_shapes
:?????????
$
_user_specified_name
N_of_Paper:QM
'
_output_shapes
:?????????
"
_user_specified_name
N_of_SCI:\X
'
_output_shapes
:?????????
-
_user_specified_nameNational_Strategy_2:PL
'
_output_shapes
:?????????
!
_user_specified_name	RnD_Org:RN
'
_output_shapes
:?????????
#
_user_specified_name	RnD_Stage:TP
'
_output_shapes
:?????????
%
_user_specified_nameSTP_Code_11:ZV
'
_output_shapes
:?????????
+
_user_specified_nameSTP_Code_1_Weight:TP
'
_output_shapes
:?????????
%
_user_specified_nameSTP_Code_21:ZV
'
_output_shapes
:?????????
+
_user_specified_nameSTP_Code_2_Weight:OK
'
_output_shapes
:?????????
 
_user_specified_nameSixT_2:MI
'
_output_shapes
:?????????

_user_specified_nameYear:

_output_shapes
: :

_output_shapes
: :!

_output_shapes
: :#

_output_shapes
: :%

_output_shapes
: :'

_output_shapes
: :)

_output_shapes
: :+

_output_shapes
: :-

_output_shapes
: :/

_output_shapes
: :1

_output_shapes
: :3

_output_shapes
: :5

_output_shapes
: :7

_output_shapes
: :9

_output_shapes
: :;

_output_shapes
: 
??
?
J__inference_dense_features_layer_call_and_return_conditional_losses_133489
features_application_area_1&
"features_application_area_1_weight
features_application_area_2&
"features_application_area_2_weight
features_cowork_abroad
features_cowork_cor
features_cowork_inst
features_cowork_uni
features_cowork_etc
features_econ_social	
features_green_tech	
features_log_duration
features_log_rnd_fund
features_multi_year	
features_n_patent_app
features_n_patent_reg
features_n_of_korean_patent
features_n_of_paper
features_n_of_sci 
features_national_strategy_2	
features_rnd_org	
features_rnd_stage	
features_stp_code_11
features_stp_code_1_weight
features_stp_code_21
features_stp_code_2_weight
features_sixt_2	
features_yearK
Gapplication_area_1_indicator_none_lookup_lookuptablefindv2_table_handleL
Happlication_area_1_indicator_none_lookup_lookuptablefindv2_default_value	K
Gapplication_area_2_indicator_none_lookup_lookuptablefindv2_table_handleL
Happlication_area_2_indicator_none_lookup_lookuptablefindv2_default_value	F
Bcowork_abroad_indicator_none_lookup_lookuptablefindv2_table_handleG
Ccowork_abroad_indicator_none_lookup_lookuptablefindv2_default_value	C
?cowork_cor_indicator_none_lookup_lookuptablefindv2_table_handleD
@cowork_cor_indicator_none_lookup_lookuptablefindv2_default_value	D
@cowork_inst_indicator_none_lookup_lookuptablefindv2_table_handleE
Acowork_inst_indicator_none_lookup_lookuptablefindv2_default_value	C
?cowork_uni_indicator_none_lookup_lookuptablefindv2_table_handleD
@cowork_uni_indicator_none_lookup_lookuptablefindv2_default_value	C
?cowork_etc_indicator_none_lookup_lookuptablefindv2_table_handleD
@cowork_etc_indicator_none_lookup_lookuptablefindv2_default_value	D
@econ_social_indicator_none_lookup_lookuptablefindv2_table_handleE
Aecon_social_indicator_none_lookup_lookuptablefindv2_default_value	C
?green_tech_indicator_none_lookup_lookuptablefindv2_table_handleD
@green_tech_indicator_none_lookup_lookuptablefindv2_default_value	C
?multi_year_indicator_none_lookup_lookuptablefindv2_table_handleD
@multi_year_indicator_none_lookup_lookuptablefindv2_default_value	L
Hnational_strategy_2_indicator_none_lookup_lookuptablefindv2_table_handleM
Inational_strategy_2_indicator_none_lookup_lookuptablefindv2_default_value	@
<rnd_org_indicator_none_lookup_lookuptablefindv2_table_handleA
=rnd_org_indicator_none_lookup_lookuptablefindv2_default_value	B
>rnd_stage_indicator_none_lookup_lookuptablefindv2_table_handleC
?rnd_stage_indicator_none_lookup_lookuptablefindv2_default_value	D
@stp_code_11_indicator_none_lookup_lookuptablefindv2_table_handleE
Astp_code_11_indicator_none_lookup_lookuptablefindv2_default_value	D
@stp_code_21_indicator_none_lookup_lookuptablefindv2_table_handleE
Astp_code_21_indicator_none_lookup_lookuptablefindv2_default_value	?
;sixt_2_indicator_none_lookup_lookuptablefindv2_table_handle@
<sixt_2_indicator_none_lookup_lookuptablefindv2_default_value	
identity??:Application_Area_1_indicator/None_Lookup/LookupTableFindV2?:Application_Area_2_indicator/None_Lookup/LookupTableFindV2?5Cowork_Abroad_indicator/None_Lookup/LookupTableFindV2?2Cowork_Cor_indicator/None_Lookup/LookupTableFindV2?3Cowork_Inst_indicator/None_Lookup/LookupTableFindV2?2Cowork_Uni_indicator/None_Lookup/LookupTableFindV2?2Cowork_etc_indicator/None_Lookup/LookupTableFindV2?3Econ_Social_indicator/None_Lookup/LookupTableFindV2?2Green_Tech_indicator/None_Lookup/LookupTableFindV2?2Multi_Year_indicator/None_Lookup/LookupTableFindV2?;National_Strategy_2_indicator/None_Lookup/LookupTableFindV2?/RnD_Org_indicator/None_Lookup/LookupTableFindV2?1RnD_Stage_indicator/None_Lookup/LookupTableFindV2?3STP_Code_11_indicator/None_Lookup/LookupTableFindV2?3STP_Code_21_indicator/None_Lookup/LookupTableFindV2?.SixT_2_indicator/None_Lookup/LookupTableFindV2q
Application_Area_1_Weight/ShapeShape"features_application_area_1_weight*
T0*
_output_shapes
:w
-Application_Area_1_Weight/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: y
/Application_Area_1_Weight/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:y
/Application_Area_1_Weight/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
'Application_Area_1_Weight/strided_sliceStridedSlice(Application_Area_1_Weight/Shape:output:06Application_Area_1_Weight/strided_slice/stack:output:08Application_Area_1_Weight/strided_slice/stack_1:output:08Application_Area_1_Weight/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskk
)Application_Area_1_Weight/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
'Application_Area_1_Weight/Reshape/shapePack0Application_Area_1_Weight/strided_slice:output:02Application_Area_1_Weight/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
!Application_Area_1_Weight/ReshapeReshape"features_application_area_1_weight0Application_Area_1_Weight/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????|
;Application_Area_1_indicator/to_sparse_input/ignore_value/xConst*
_output_shapes
: *
dtype0*
valueB B ?
5Application_Area_1_indicator/to_sparse_input/NotEqualNotEqualfeatures_application_area_1DApplication_Area_1_indicator/to_sparse_input/ignore_value/x:output:0*
T0*'
_output_shapes
:??????????
4Application_Area_1_indicator/to_sparse_input/indicesWhere9Application_Area_1_indicator/to_sparse_input/NotEqual:z:0*'
_output_shapes
:??????????
3Application_Area_1_indicator/to_sparse_input/valuesGatherNdfeatures_application_area_1<Application_Area_1_indicator/to_sparse_input/indices:index:0*
Tindices0	*
Tparams0*#
_output_shapes
:??????????
8Application_Area_1_indicator/to_sparse_input/dense_shapeShapefeatures_application_area_1*
T0*
_output_shapes
:*
out_type0	?
:Application_Area_1_indicator/None_Lookup/LookupTableFindV2LookupTableFindV2Gapplication_area_1_indicator_none_lookup_lookuptablefindv2_table_handle<Application_Area_1_indicator/to_sparse_input/values:output:0Happlication_area_1_indicator_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:??????????
8Application_Area_1_indicator/SparseToDense/default_valueConst*
_output_shapes
: *
dtype0	*
valueB	 R
??????????
*Application_Area_1_indicator/SparseToDenseSparseToDense<Application_Area_1_indicator/to_sparse_input/indices:index:0AApplication_Area_1_indicator/to_sparse_input/dense_shape:output:0CApplication_Area_1_indicator/None_Lookup/LookupTableFindV2:values:0AApplication_Area_1_indicator/SparseToDense/default_value:output:0*
T0	*
Tindices0	*'
_output_shapes
:?????????o
*Application_Area_1_indicator/one_hot/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??q
,Application_Area_1_indicator/one_hot/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    l
*Application_Area_1_indicator/one_hot/depthConst*
_output_shapes
: *
dtype0*
value	B :!?
$Application_Area_1_indicator/one_hotOneHot2Application_Area_1_indicator/SparseToDense:dense:03Application_Area_1_indicator/one_hot/depth:output:03Application_Area_1_indicator/one_hot/Const:output:05Application_Area_1_indicator/one_hot/Const_1:output:0*
T0*+
_output_shapes
:?????????!?
2Application_Area_1_indicator/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
??????????
 Application_Area_1_indicator/SumSum-Application_Area_1_indicator/one_hot:output:0;Application_Area_1_indicator/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:?????????!{
"Application_Area_1_indicator/ShapeShape)Application_Area_1_indicator/Sum:output:0*
T0*
_output_shapes
:z
0Application_Area_1_indicator/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: |
2Application_Area_1_indicator/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:|
2Application_Area_1_indicator/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
*Application_Area_1_indicator/strided_sliceStridedSlice+Application_Area_1_indicator/Shape:output:09Application_Area_1_indicator/strided_slice/stack:output:0;Application_Area_1_indicator/strided_slice/stack_1:output:0;Application_Area_1_indicator/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskn
,Application_Area_1_indicator/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :!?
*Application_Area_1_indicator/Reshape/shapePack3Application_Area_1_indicator/strided_slice:output:05Application_Area_1_indicator/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
$Application_Area_1_indicator/ReshapeReshape)Application_Area_1_indicator/Sum:output:03Application_Area_1_indicator/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????!q
Application_Area_2_Weight/ShapeShape"features_application_area_2_weight*
T0*
_output_shapes
:w
-Application_Area_2_Weight/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: y
/Application_Area_2_Weight/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:y
/Application_Area_2_Weight/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
'Application_Area_2_Weight/strided_sliceStridedSlice(Application_Area_2_Weight/Shape:output:06Application_Area_2_Weight/strided_slice/stack:output:08Application_Area_2_Weight/strided_slice/stack_1:output:08Application_Area_2_Weight/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskk
)Application_Area_2_Weight/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
'Application_Area_2_Weight/Reshape/shapePack0Application_Area_2_Weight/strided_slice:output:02Application_Area_2_Weight/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
!Application_Area_2_Weight/ReshapeReshape"features_application_area_2_weight0Application_Area_2_Weight/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????|
;Application_Area_2_indicator/to_sparse_input/ignore_value/xConst*
_output_shapes
: *
dtype0*
valueB B ?
5Application_Area_2_indicator/to_sparse_input/NotEqualNotEqualfeatures_application_area_2DApplication_Area_2_indicator/to_sparse_input/ignore_value/x:output:0*
T0*'
_output_shapes
:??????????
4Application_Area_2_indicator/to_sparse_input/indicesWhere9Application_Area_2_indicator/to_sparse_input/NotEqual:z:0*'
_output_shapes
:??????????
3Application_Area_2_indicator/to_sparse_input/valuesGatherNdfeatures_application_area_2<Application_Area_2_indicator/to_sparse_input/indices:index:0*
Tindices0	*
Tparams0*#
_output_shapes
:??????????
8Application_Area_2_indicator/to_sparse_input/dense_shapeShapefeatures_application_area_2*
T0*
_output_shapes
:*
out_type0	?
:Application_Area_2_indicator/None_Lookup/LookupTableFindV2LookupTableFindV2Gapplication_area_2_indicator_none_lookup_lookuptablefindv2_table_handle<Application_Area_2_indicator/to_sparse_input/values:output:0Happlication_area_2_indicator_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:??????????
8Application_Area_2_indicator/SparseToDense/default_valueConst*
_output_shapes
: *
dtype0	*
valueB	 R
??????????
*Application_Area_2_indicator/SparseToDenseSparseToDense<Application_Area_2_indicator/to_sparse_input/indices:index:0AApplication_Area_2_indicator/to_sparse_input/dense_shape:output:0CApplication_Area_2_indicator/None_Lookup/LookupTableFindV2:values:0AApplication_Area_2_indicator/SparseToDense/default_value:output:0*
T0	*
Tindices0	*'
_output_shapes
:?????????o
*Application_Area_2_indicator/one_hot/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??q
,Application_Area_2_indicator/one_hot/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    l
*Application_Area_2_indicator/one_hot/depthConst*
_output_shapes
: *
dtype0*
value	B :"?
$Application_Area_2_indicator/one_hotOneHot2Application_Area_2_indicator/SparseToDense:dense:03Application_Area_2_indicator/one_hot/depth:output:03Application_Area_2_indicator/one_hot/Const:output:05Application_Area_2_indicator/one_hot/Const_1:output:0*
T0*+
_output_shapes
:?????????"?
2Application_Area_2_indicator/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
??????????
 Application_Area_2_indicator/SumSum-Application_Area_2_indicator/one_hot:output:0;Application_Area_2_indicator/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:?????????"{
"Application_Area_2_indicator/ShapeShape)Application_Area_2_indicator/Sum:output:0*
T0*
_output_shapes
:z
0Application_Area_2_indicator/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: |
2Application_Area_2_indicator/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:|
2Application_Area_2_indicator/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
*Application_Area_2_indicator/strided_sliceStridedSlice+Application_Area_2_indicator/Shape:output:09Application_Area_2_indicator/strided_slice/stack:output:0;Application_Area_2_indicator/strided_slice/stack_1:output:0;Application_Area_2_indicator/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskn
,Application_Area_2_indicator/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :"?
*Application_Area_2_indicator/Reshape/shapePack3Application_Area_2_indicator/strided_slice:output:05Application_Area_2_indicator/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
$Application_Area_2_indicator/ReshapeReshape)Application_Area_2_indicator/Sum:output:03Application_Area_2_indicator/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????"w
6Cowork_Abroad_indicator/to_sparse_input/ignore_value/xConst*
_output_shapes
: *
dtype0*
valueB B ?
0Cowork_Abroad_indicator/to_sparse_input/NotEqualNotEqualfeatures_cowork_abroad?Cowork_Abroad_indicator/to_sparse_input/ignore_value/x:output:0*
T0*'
_output_shapes
:??????????
/Cowork_Abroad_indicator/to_sparse_input/indicesWhere4Cowork_Abroad_indicator/to_sparse_input/NotEqual:z:0*'
_output_shapes
:??????????
.Cowork_Abroad_indicator/to_sparse_input/valuesGatherNdfeatures_cowork_abroad7Cowork_Abroad_indicator/to_sparse_input/indices:index:0*
Tindices0	*
Tparams0*#
_output_shapes
:??????????
3Cowork_Abroad_indicator/to_sparse_input/dense_shapeShapefeatures_cowork_abroad*
T0*
_output_shapes
:*
out_type0	?
5Cowork_Abroad_indicator/None_Lookup/LookupTableFindV2LookupTableFindV2Bcowork_abroad_indicator_none_lookup_lookuptablefindv2_table_handle7Cowork_Abroad_indicator/to_sparse_input/values:output:0Ccowork_abroad_indicator_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:?????????~
3Cowork_Abroad_indicator/SparseToDense/default_valueConst*
_output_shapes
: *
dtype0	*
valueB	 R
??????????
%Cowork_Abroad_indicator/SparseToDenseSparseToDense7Cowork_Abroad_indicator/to_sparse_input/indices:index:0<Cowork_Abroad_indicator/to_sparse_input/dense_shape:output:0>Cowork_Abroad_indicator/None_Lookup/LookupTableFindV2:values:0<Cowork_Abroad_indicator/SparseToDense/default_value:output:0*
T0	*
Tindices0	*'
_output_shapes
:?????????j
%Cowork_Abroad_indicator/one_hot/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??l
'Cowork_Abroad_indicator/one_hot/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    g
%Cowork_Abroad_indicator/one_hot/depthConst*
_output_shapes
: *
dtype0*
value	B :?
Cowork_Abroad_indicator/one_hotOneHot-Cowork_Abroad_indicator/SparseToDense:dense:0.Cowork_Abroad_indicator/one_hot/depth:output:0.Cowork_Abroad_indicator/one_hot/Const:output:00Cowork_Abroad_indicator/one_hot/Const_1:output:0*
T0*+
_output_shapes
:??????????
-Cowork_Abroad_indicator/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
??????????
Cowork_Abroad_indicator/SumSum(Cowork_Abroad_indicator/one_hot:output:06Cowork_Abroad_indicator/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:?????????q
Cowork_Abroad_indicator/ShapeShape$Cowork_Abroad_indicator/Sum:output:0*
T0*
_output_shapes
:u
+Cowork_Abroad_indicator/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: w
-Cowork_Abroad_indicator/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:w
-Cowork_Abroad_indicator/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
%Cowork_Abroad_indicator/strided_sliceStridedSlice&Cowork_Abroad_indicator/Shape:output:04Cowork_Abroad_indicator/strided_slice/stack:output:06Cowork_Abroad_indicator/strided_slice/stack_1:output:06Cowork_Abroad_indicator/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maski
'Cowork_Abroad_indicator/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
%Cowork_Abroad_indicator/Reshape/shapePack.Cowork_Abroad_indicator/strided_slice:output:00Cowork_Abroad_indicator/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
Cowork_Abroad_indicator/ReshapeReshape$Cowork_Abroad_indicator/Sum:output:0.Cowork_Abroad_indicator/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????t
3Cowork_Cor_indicator/to_sparse_input/ignore_value/xConst*
_output_shapes
: *
dtype0*
valueB B ?
-Cowork_Cor_indicator/to_sparse_input/NotEqualNotEqualfeatures_cowork_cor<Cowork_Cor_indicator/to_sparse_input/ignore_value/x:output:0*
T0*'
_output_shapes
:??????????
,Cowork_Cor_indicator/to_sparse_input/indicesWhere1Cowork_Cor_indicator/to_sparse_input/NotEqual:z:0*'
_output_shapes
:??????????
+Cowork_Cor_indicator/to_sparse_input/valuesGatherNdfeatures_cowork_cor4Cowork_Cor_indicator/to_sparse_input/indices:index:0*
Tindices0	*
Tparams0*#
_output_shapes
:??????????
0Cowork_Cor_indicator/to_sparse_input/dense_shapeShapefeatures_cowork_cor*
T0*
_output_shapes
:*
out_type0	?
2Cowork_Cor_indicator/None_Lookup/LookupTableFindV2LookupTableFindV2?cowork_cor_indicator_none_lookup_lookuptablefindv2_table_handle4Cowork_Cor_indicator/to_sparse_input/values:output:0@cowork_cor_indicator_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:?????????{
0Cowork_Cor_indicator/SparseToDense/default_valueConst*
_output_shapes
: *
dtype0	*
valueB	 R
??????????
"Cowork_Cor_indicator/SparseToDenseSparseToDense4Cowork_Cor_indicator/to_sparse_input/indices:index:09Cowork_Cor_indicator/to_sparse_input/dense_shape:output:0;Cowork_Cor_indicator/None_Lookup/LookupTableFindV2:values:09Cowork_Cor_indicator/SparseToDense/default_value:output:0*
T0	*
Tindices0	*'
_output_shapes
:?????????g
"Cowork_Cor_indicator/one_hot/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??i
$Cowork_Cor_indicator/one_hot/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    d
"Cowork_Cor_indicator/one_hot/depthConst*
_output_shapes
: *
dtype0*
value	B :?
Cowork_Cor_indicator/one_hotOneHot*Cowork_Cor_indicator/SparseToDense:dense:0+Cowork_Cor_indicator/one_hot/depth:output:0+Cowork_Cor_indicator/one_hot/Const:output:0-Cowork_Cor_indicator/one_hot/Const_1:output:0*
T0*+
_output_shapes
:?????????}
*Cowork_Cor_indicator/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
??????????
Cowork_Cor_indicator/SumSum%Cowork_Cor_indicator/one_hot:output:03Cowork_Cor_indicator/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:?????????k
Cowork_Cor_indicator/ShapeShape!Cowork_Cor_indicator/Sum:output:0*
T0*
_output_shapes
:r
(Cowork_Cor_indicator/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: t
*Cowork_Cor_indicator/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:t
*Cowork_Cor_indicator/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
"Cowork_Cor_indicator/strided_sliceStridedSlice#Cowork_Cor_indicator/Shape:output:01Cowork_Cor_indicator/strided_slice/stack:output:03Cowork_Cor_indicator/strided_slice/stack_1:output:03Cowork_Cor_indicator/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
$Cowork_Cor_indicator/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
"Cowork_Cor_indicator/Reshape/shapePack+Cowork_Cor_indicator/strided_slice:output:0-Cowork_Cor_indicator/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
Cowork_Cor_indicator/ReshapeReshape!Cowork_Cor_indicator/Sum:output:0+Cowork_Cor_indicator/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????u
4Cowork_Inst_indicator/to_sparse_input/ignore_value/xConst*
_output_shapes
: *
dtype0*
valueB B ?
.Cowork_Inst_indicator/to_sparse_input/NotEqualNotEqualfeatures_cowork_inst=Cowork_Inst_indicator/to_sparse_input/ignore_value/x:output:0*
T0*'
_output_shapes
:??????????
-Cowork_Inst_indicator/to_sparse_input/indicesWhere2Cowork_Inst_indicator/to_sparse_input/NotEqual:z:0*'
_output_shapes
:??????????
,Cowork_Inst_indicator/to_sparse_input/valuesGatherNdfeatures_cowork_inst5Cowork_Inst_indicator/to_sparse_input/indices:index:0*
Tindices0	*
Tparams0*#
_output_shapes
:??????????
1Cowork_Inst_indicator/to_sparse_input/dense_shapeShapefeatures_cowork_inst*
T0*
_output_shapes
:*
out_type0	?
3Cowork_Inst_indicator/None_Lookup/LookupTableFindV2LookupTableFindV2@cowork_inst_indicator_none_lookup_lookuptablefindv2_table_handle5Cowork_Inst_indicator/to_sparse_input/values:output:0Acowork_inst_indicator_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:?????????|
1Cowork_Inst_indicator/SparseToDense/default_valueConst*
_output_shapes
: *
dtype0	*
valueB	 R
??????????
#Cowork_Inst_indicator/SparseToDenseSparseToDense5Cowork_Inst_indicator/to_sparse_input/indices:index:0:Cowork_Inst_indicator/to_sparse_input/dense_shape:output:0<Cowork_Inst_indicator/None_Lookup/LookupTableFindV2:values:0:Cowork_Inst_indicator/SparseToDense/default_value:output:0*
T0	*
Tindices0	*'
_output_shapes
:?????????h
#Cowork_Inst_indicator/one_hot/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??j
%Cowork_Inst_indicator/one_hot/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    e
#Cowork_Inst_indicator/one_hot/depthConst*
_output_shapes
: *
dtype0*
value	B :?
Cowork_Inst_indicator/one_hotOneHot+Cowork_Inst_indicator/SparseToDense:dense:0,Cowork_Inst_indicator/one_hot/depth:output:0,Cowork_Inst_indicator/one_hot/Const:output:0.Cowork_Inst_indicator/one_hot/Const_1:output:0*
T0*+
_output_shapes
:?????????~
+Cowork_Inst_indicator/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
??????????
Cowork_Inst_indicator/SumSum&Cowork_Inst_indicator/one_hot:output:04Cowork_Inst_indicator/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:?????????m
Cowork_Inst_indicator/ShapeShape"Cowork_Inst_indicator/Sum:output:0*
T0*
_output_shapes
:s
)Cowork_Inst_indicator/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: u
+Cowork_Inst_indicator/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+Cowork_Inst_indicator/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
#Cowork_Inst_indicator/strided_sliceStridedSlice$Cowork_Inst_indicator/Shape:output:02Cowork_Inst_indicator/strided_slice/stack:output:04Cowork_Inst_indicator/strided_slice/stack_1:output:04Cowork_Inst_indicator/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskg
%Cowork_Inst_indicator/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
#Cowork_Inst_indicator/Reshape/shapePack,Cowork_Inst_indicator/strided_slice:output:0.Cowork_Inst_indicator/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
Cowork_Inst_indicator/ReshapeReshape"Cowork_Inst_indicator/Sum:output:0,Cowork_Inst_indicator/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????t
3Cowork_Uni_indicator/to_sparse_input/ignore_value/xConst*
_output_shapes
: *
dtype0*
valueB B ?
-Cowork_Uni_indicator/to_sparse_input/NotEqualNotEqualfeatures_cowork_uni<Cowork_Uni_indicator/to_sparse_input/ignore_value/x:output:0*
T0*'
_output_shapes
:??????????
,Cowork_Uni_indicator/to_sparse_input/indicesWhere1Cowork_Uni_indicator/to_sparse_input/NotEqual:z:0*'
_output_shapes
:??????????
+Cowork_Uni_indicator/to_sparse_input/valuesGatherNdfeatures_cowork_uni4Cowork_Uni_indicator/to_sparse_input/indices:index:0*
Tindices0	*
Tparams0*#
_output_shapes
:??????????
0Cowork_Uni_indicator/to_sparse_input/dense_shapeShapefeatures_cowork_uni*
T0*
_output_shapes
:*
out_type0	?
2Cowork_Uni_indicator/None_Lookup/LookupTableFindV2LookupTableFindV2?cowork_uni_indicator_none_lookup_lookuptablefindv2_table_handle4Cowork_Uni_indicator/to_sparse_input/values:output:0@cowork_uni_indicator_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:?????????{
0Cowork_Uni_indicator/SparseToDense/default_valueConst*
_output_shapes
: *
dtype0	*
valueB	 R
??????????
"Cowork_Uni_indicator/SparseToDenseSparseToDense4Cowork_Uni_indicator/to_sparse_input/indices:index:09Cowork_Uni_indicator/to_sparse_input/dense_shape:output:0;Cowork_Uni_indicator/None_Lookup/LookupTableFindV2:values:09Cowork_Uni_indicator/SparseToDense/default_value:output:0*
T0	*
Tindices0	*'
_output_shapes
:?????????g
"Cowork_Uni_indicator/one_hot/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??i
$Cowork_Uni_indicator/one_hot/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    d
"Cowork_Uni_indicator/one_hot/depthConst*
_output_shapes
: *
dtype0*
value	B :?
Cowork_Uni_indicator/one_hotOneHot*Cowork_Uni_indicator/SparseToDense:dense:0+Cowork_Uni_indicator/one_hot/depth:output:0+Cowork_Uni_indicator/one_hot/Const:output:0-Cowork_Uni_indicator/one_hot/Const_1:output:0*
T0*+
_output_shapes
:?????????}
*Cowork_Uni_indicator/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
??????????
Cowork_Uni_indicator/SumSum%Cowork_Uni_indicator/one_hot:output:03Cowork_Uni_indicator/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:?????????k
Cowork_Uni_indicator/ShapeShape!Cowork_Uni_indicator/Sum:output:0*
T0*
_output_shapes
:r
(Cowork_Uni_indicator/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: t
*Cowork_Uni_indicator/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:t
*Cowork_Uni_indicator/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
"Cowork_Uni_indicator/strided_sliceStridedSlice#Cowork_Uni_indicator/Shape:output:01Cowork_Uni_indicator/strided_slice/stack:output:03Cowork_Uni_indicator/strided_slice/stack_1:output:03Cowork_Uni_indicator/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
$Cowork_Uni_indicator/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
"Cowork_Uni_indicator/Reshape/shapePack+Cowork_Uni_indicator/strided_slice:output:0-Cowork_Uni_indicator/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
Cowork_Uni_indicator/ReshapeReshape!Cowork_Uni_indicator/Sum:output:0+Cowork_Uni_indicator/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????t
3Cowork_etc_indicator/to_sparse_input/ignore_value/xConst*
_output_shapes
: *
dtype0*
valueB B ?
-Cowork_etc_indicator/to_sparse_input/NotEqualNotEqualfeatures_cowork_etc<Cowork_etc_indicator/to_sparse_input/ignore_value/x:output:0*
T0*'
_output_shapes
:??????????
,Cowork_etc_indicator/to_sparse_input/indicesWhere1Cowork_etc_indicator/to_sparse_input/NotEqual:z:0*'
_output_shapes
:??????????
+Cowork_etc_indicator/to_sparse_input/valuesGatherNdfeatures_cowork_etc4Cowork_etc_indicator/to_sparse_input/indices:index:0*
Tindices0	*
Tparams0*#
_output_shapes
:??????????
0Cowork_etc_indicator/to_sparse_input/dense_shapeShapefeatures_cowork_etc*
T0*
_output_shapes
:*
out_type0	?
2Cowork_etc_indicator/None_Lookup/LookupTableFindV2LookupTableFindV2?cowork_etc_indicator_none_lookup_lookuptablefindv2_table_handle4Cowork_etc_indicator/to_sparse_input/values:output:0@cowork_etc_indicator_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:?????????{
0Cowork_etc_indicator/SparseToDense/default_valueConst*
_output_shapes
: *
dtype0	*
valueB	 R
??????????
"Cowork_etc_indicator/SparseToDenseSparseToDense4Cowork_etc_indicator/to_sparse_input/indices:index:09Cowork_etc_indicator/to_sparse_input/dense_shape:output:0;Cowork_etc_indicator/None_Lookup/LookupTableFindV2:values:09Cowork_etc_indicator/SparseToDense/default_value:output:0*
T0	*
Tindices0	*'
_output_shapes
:?????????g
"Cowork_etc_indicator/one_hot/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??i
$Cowork_etc_indicator/one_hot/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    d
"Cowork_etc_indicator/one_hot/depthConst*
_output_shapes
: *
dtype0*
value	B :?
Cowork_etc_indicator/one_hotOneHot*Cowork_etc_indicator/SparseToDense:dense:0+Cowork_etc_indicator/one_hot/depth:output:0+Cowork_etc_indicator/one_hot/Const:output:0-Cowork_etc_indicator/one_hot/Const_1:output:0*
T0*+
_output_shapes
:?????????}
*Cowork_etc_indicator/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
??????????
Cowork_etc_indicator/SumSum%Cowork_etc_indicator/one_hot:output:03Cowork_etc_indicator/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:?????????k
Cowork_etc_indicator/ShapeShape!Cowork_etc_indicator/Sum:output:0*
T0*
_output_shapes
:r
(Cowork_etc_indicator/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: t
*Cowork_etc_indicator/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:t
*Cowork_etc_indicator/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
"Cowork_etc_indicator/strided_sliceStridedSlice#Cowork_etc_indicator/Shape:output:01Cowork_etc_indicator/strided_slice/stack:output:03Cowork_etc_indicator/strided_slice/stack_1:output:03Cowork_etc_indicator/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
$Cowork_etc_indicator/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
"Cowork_etc_indicator/Reshape/shapePack+Cowork_etc_indicator/strided_slice:output:0-Cowork_etc_indicator/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
Cowork_etc_indicator/ReshapeReshape!Cowork_etc_indicator/Sum:output:0+Cowork_etc_indicator/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????
4Econ_Social_indicator/to_sparse_input/ignore_value/xConst*
_output_shapes
: *
dtype0*
valueB :
??????????
2Econ_Social_indicator/to_sparse_input/ignore_valueCast=Econ_Social_indicator/to_sparse_input/ignore_value/x:output:0*

DstT0	*

SrcT0*
_output_shapes
: ?
.Econ_Social_indicator/to_sparse_input/NotEqualNotEqualfeatures_econ_social6Econ_Social_indicator/to_sparse_input/ignore_value:y:0*
T0	*'
_output_shapes
:??????????
-Econ_Social_indicator/to_sparse_input/indicesWhere2Econ_Social_indicator/to_sparse_input/NotEqual:z:0*'
_output_shapes
:??????????
,Econ_Social_indicator/to_sparse_input/valuesGatherNdfeatures_econ_social5Econ_Social_indicator/to_sparse_input/indices:index:0*
Tindices0	*
Tparams0	*#
_output_shapes
:??????????
1Econ_Social_indicator/to_sparse_input/dense_shapeShapefeatures_econ_social*
T0	*
_output_shapes
:*
out_type0	?
3Econ_Social_indicator/None_Lookup/LookupTableFindV2LookupTableFindV2@econ_social_indicator_none_lookup_lookuptablefindv2_table_handle5Econ_Social_indicator/to_sparse_input/values:output:0Aecon_social_indicator_none_lookup_lookuptablefindv2_default_value*	
Tin0	*

Tout0	*#
_output_shapes
:?????????|
1Econ_Social_indicator/SparseToDense/default_valueConst*
_output_shapes
: *
dtype0	*
valueB	 R
??????????
#Econ_Social_indicator/SparseToDenseSparseToDense5Econ_Social_indicator/to_sparse_input/indices:index:0:Econ_Social_indicator/to_sparse_input/dense_shape:output:0<Econ_Social_indicator/None_Lookup/LookupTableFindV2:values:0:Econ_Social_indicator/SparseToDense/default_value:output:0*
T0	*
Tindices0	*'
_output_shapes
:?????????h
#Econ_Social_indicator/one_hot/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??j
%Econ_Social_indicator/one_hot/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    e
#Econ_Social_indicator/one_hot/depthConst*
_output_shapes
: *
dtype0*
value	B :?
Econ_Social_indicator/one_hotOneHot+Econ_Social_indicator/SparseToDense:dense:0,Econ_Social_indicator/one_hot/depth:output:0,Econ_Social_indicator/one_hot/Const:output:0.Econ_Social_indicator/one_hot/Const_1:output:0*
T0*+
_output_shapes
:?????????~
+Econ_Social_indicator/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
??????????
Econ_Social_indicator/SumSum&Econ_Social_indicator/one_hot:output:04Econ_Social_indicator/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:?????????m
Econ_Social_indicator/ShapeShape"Econ_Social_indicator/Sum:output:0*
T0*
_output_shapes
:s
)Econ_Social_indicator/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: u
+Econ_Social_indicator/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+Econ_Social_indicator/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
#Econ_Social_indicator/strided_sliceStridedSlice$Econ_Social_indicator/Shape:output:02Econ_Social_indicator/strided_slice/stack:output:04Econ_Social_indicator/strided_slice/stack_1:output:04Econ_Social_indicator/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskg
%Econ_Social_indicator/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
#Econ_Social_indicator/Reshape/shapePack,Econ_Social_indicator/strided_slice:output:0.Econ_Social_indicator/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
Econ_Social_indicator/ReshapeReshape"Econ_Social_indicator/Sum:output:0,Econ_Social_indicator/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????~
3Green_Tech_indicator/to_sparse_input/ignore_value/xConst*
_output_shapes
: *
dtype0*
valueB :
??????????
1Green_Tech_indicator/to_sparse_input/ignore_valueCast<Green_Tech_indicator/to_sparse_input/ignore_value/x:output:0*

DstT0	*

SrcT0*
_output_shapes
: ?
-Green_Tech_indicator/to_sparse_input/NotEqualNotEqualfeatures_green_tech5Green_Tech_indicator/to_sparse_input/ignore_value:y:0*
T0	*'
_output_shapes
:??????????
,Green_Tech_indicator/to_sparse_input/indicesWhere1Green_Tech_indicator/to_sparse_input/NotEqual:z:0*'
_output_shapes
:??????????
+Green_Tech_indicator/to_sparse_input/valuesGatherNdfeatures_green_tech4Green_Tech_indicator/to_sparse_input/indices:index:0*
Tindices0	*
Tparams0	*#
_output_shapes
:??????????
0Green_Tech_indicator/to_sparse_input/dense_shapeShapefeatures_green_tech*
T0	*
_output_shapes
:*
out_type0	?
2Green_Tech_indicator/None_Lookup/LookupTableFindV2LookupTableFindV2?green_tech_indicator_none_lookup_lookuptablefindv2_table_handle4Green_Tech_indicator/to_sparse_input/values:output:0@green_tech_indicator_none_lookup_lookuptablefindv2_default_value*	
Tin0	*

Tout0	*#
_output_shapes
:?????????{
0Green_Tech_indicator/SparseToDense/default_valueConst*
_output_shapes
: *
dtype0	*
valueB	 R
??????????
"Green_Tech_indicator/SparseToDenseSparseToDense4Green_Tech_indicator/to_sparse_input/indices:index:09Green_Tech_indicator/to_sparse_input/dense_shape:output:0;Green_Tech_indicator/None_Lookup/LookupTableFindV2:values:09Green_Tech_indicator/SparseToDense/default_value:output:0*
T0	*
Tindices0	*'
_output_shapes
:?????????g
"Green_Tech_indicator/one_hot/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??i
$Green_Tech_indicator/one_hot/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    d
"Green_Tech_indicator/one_hot/depthConst*
_output_shapes
: *
dtype0*
value	B :<?
Green_Tech_indicator/one_hotOneHot*Green_Tech_indicator/SparseToDense:dense:0+Green_Tech_indicator/one_hot/depth:output:0+Green_Tech_indicator/one_hot/Const:output:0-Green_Tech_indicator/one_hot/Const_1:output:0*
T0*+
_output_shapes
:?????????<}
*Green_Tech_indicator/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
??????????
Green_Tech_indicator/SumSum%Green_Tech_indicator/one_hot:output:03Green_Tech_indicator/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:?????????<k
Green_Tech_indicator/ShapeShape!Green_Tech_indicator/Sum:output:0*
T0*
_output_shapes
:r
(Green_Tech_indicator/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: t
*Green_Tech_indicator/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:t
*Green_Tech_indicator/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
"Green_Tech_indicator/strided_sliceStridedSlice#Green_Tech_indicator/Shape:output:01Green_Tech_indicator/strided_slice/stack:output:03Green_Tech_indicator/strided_slice/stack_1:output:03Green_Tech_indicator/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
$Green_Tech_indicator/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :<?
"Green_Tech_indicator/Reshape/shapePack+Green_Tech_indicator/strided_slice:output:0-Green_Tech_indicator/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
Green_Tech_indicator/ReshapeReshape!Green_Tech_indicator/Sum:output:0+Green_Tech_indicator/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????<W
Log_Duration/ShapeShapefeatures_log_duration*
T0*
_output_shapes
:j
 Log_Duration/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: l
"Log_Duration/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:l
"Log_Duration/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
Log_Duration/strided_sliceStridedSliceLog_Duration/Shape:output:0)Log_Duration/strided_slice/stack:output:0+Log_Duration/strided_slice/stack_1:output:0+Log_Duration/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask^
Log_Duration/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
Log_Duration/Reshape/shapePack#Log_Duration/strided_slice:output:0%Log_Duration/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
Log_Duration/ReshapeReshapefeatures_log_duration#Log_Duration/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????W
Log_RnD_Fund/ShapeShapefeatures_log_rnd_fund*
T0*
_output_shapes
:j
 Log_RnD_Fund/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: l
"Log_RnD_Fund/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:l
"Log_RnD_Fund/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
Log_RnD_Fund/strided_sliceStridedSliceLog_RnD_Fund/Shape:output:0)Log_RnD_Fund/strided_slice/stack:output:0+Log_RnD_Fund/strided_slice/stack_1:output:0+Log_RnD_Fund/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask^
Log_RnD_Fund/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
Log_RnD_Fund/Reshape/shapePack#Log_RnD_Fund/strided_slice:output:0%Log_RnD_Fund/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
Log_RnD_Fund/ReshapeReshapefeatures_log_rnd_fund#Log_RnD_Fund/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????~
3Multi_Year_indicator/to_sparse_input/ignore_value/xConst*
_output_shapes
: *
dtype0*
valueB :
??????????
1Multi_Year_indicator/to_sparse_input/ignore_valueCast<Multi_Year_indicator/to_sparse_input/ignore_value/x:output:0*

DstT0	*

SrcT0*
_output_shapes
: ?
-Multi_Year_indicator/to_sparse_input/NotEqualNotEqualfeatures_multi_year5Multi_Year_indicator/to_sparse_input/ignore_value:y:0*
T0	*'
_output_shapes
:??????????
,Multi_Year_indicator/to_sparse_input/indicesWhere1Multi_Year_indicator/to_sparse_input/NotEqual:z:0*'
_output_shapes
:??????????
+Multi_Year_indicator/to_sparse_input/valuesGatherNdfeatures_multi_year4Multi_Year_indicator/to_sparse_input/indices:index:0*
Tindices0	*
Tparams0	*#
_output_shapes
:??????????
0Multi_Year_indicator/to_sparse_input/dense_shapeShapefeatures_multi_year*
T0	*
_output_shapes
:*
out_type0	?
2Multi_Year_indicator/None_Lookup/LookupTableFindV2LookupTableFindV2?multi_year_indicator_none_lookup_lookuptablefindv2_table_handle4Multi_Year_indicator/to_sparse_input/values:output:0@multi_year_indicator_none_lookup_lookuptablefindv2_default_value*	
Tin0	*

Tout0	*#
_output_shapes
:?????????{
0Multi_Year_indicator/SparseToDense/default_valueConst*
_output_shapes
: *
dtype0	*
valueB	 R
??????????
"Multi_Year_indicator/SparseToDenseSparseToDense4Multi_Year_indicator/to_sparse_input/indices:index:09Multi_Year_indicator/to_sparse_input/dense_shape:output:0;Multi_Year_indicator/None_Lookup/LookupTableFindV2:values:09Multi_Year_indicator/SparseToDense/default_value:output:0*
T0	*
Tindices0	*'
_output_shapes
:?????????g
"Multi_Year_indicator/one_hot/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??i
$Multi_Year_indicator/one_hot/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    d
"Multi_Year_indicator/one_hot/depthConst*
_output_shapes
: *
dtype0*
value	B :?
Multi_Year_indicator/one_hotOneHot*Multi_Year_indicator/SparseToDense:dense:0+Multi_Year_indicator/one_hot/depth:output:0+Multi_Year_indicator/one_hot/Const:output:0-Multi_Year_indicator/one_hot/Const_1:output:0*
T0*+
_output_shapes
:?????????}
*Multi_Year_indicator/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
??????????
Multi_Year_indicator/SumSum%Multi_Year_indicator/one_hot:output:03Multi_Year_indicator/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:?????????k
Multi_Year_indicator/ShapeShape!Multi_Year_indicator/Sum:output:0*
T0*
_output_shapes
:r
(Multi_Year_indicator/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: t
*Multi_Year_indicator/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:t
*Multi_Year_indicator/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
"Multi_Year_indicator/strided_sliceStridedSlice#Multi_Year_indicator/Shape:output:01Multi_Year_indicator/strided_slice/stack:output:03Multi_Year_indicator/strided_slice/stack_1:output:03Multi_Year_indicator/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
$Multi_Year_indicator/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
"Multi_Year_indicator/Reshape/shapePack+Multi_Year_indicator/strided_slice:output:0-Multi_Year_indicator/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
Multi_Year_indicator/ReshapeReshape!Multi_Year_indicator/Sum:output:0+Multi_Year_indicator/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????W
N_Patent_App/ShapeShapefeatures_n_patent_app*
T0*
_output_shapes
:j
 N_Patent_App/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: l
"N_Patent_App/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:l
"N_Patent_App/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
N_Patent_App/strided_sliceStridedSliceN_Patent_App/Shape:output:0)N_Patent_App/strided_slice/stack:output:0+N_Patent_App/strided_slice/stack_1:output:0+N_Patent_App/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask^
N_Patent_App/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
N_Patent_App/Reshape/shapePack#N_Patent_App/strided_slice:output:0%N_Patent_App/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
N_Patent_App/ReshapeReshapefeatures_n_patent_app#N_Patent_App/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????W
N_Patent_Reg/ShapeShapefeatures_n_patent_reg*
T0*
_output_shapes
:j
 N_Patent_Reg/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: l
"N_Patent_Reg/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:l
"N_Patent_Reg/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
N_Patent_Reg/strided_sliceStridedSliceN_Patent_Reg/Shape:output:0)N_Patent_Reg/strided_slice/stack:output:0+N_Patent_Reg/strided_slice/stack_1:output:0+N_Patent_Reg/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask^
N_Patent_Reg/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
N_Patent_Reg/Reshape/shapePack#N_Patent_Reg/strided_slice:output:0%N_Patent_Reg/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
N_Patent_Reg/ReshapeReshapefeatures_n_patent_reg#N_Patent_Reg/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????c
N_of_Korean_Patent/ShapeShapefeatures_n_of_korean_patent*
T0*
_output_shapes
:p
&N_of_Korean_Patent/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: r
(N_of_Korean_Patent/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(N_of_Korean_Patent/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
 N_of_Korean_Patent/strided_sliceStridedSlice!N_of_Korean_Patent/Shape:output:0/N_of_Korean_Patent/strided_slice/stack:output:01N_of_Korean_Patent/strided_slice/stack_1:output:01N_of_Korean_Patent/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskd
"N_of_Korean_Patent/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
 N_of_Korean_Patent/Reshape/shapePack)N_of_Korean_Patent/strided_slice:output:0+N_of_Korean_Patent/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
N_of_Korean_Patent/ReshapeReshapefeatures_n_of_korean_patent)N_of_Korean_Patent/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????S
N_of_Paper/ShapeShapefeatures_n_of_paper*
T0*
_output_shapes
:h
N_of_Paper/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: j
 N_of_Paper/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:j
 N_of_Paper/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
N_of_Paper/strided_sliceStridedSliceN_of_Paper/Shape:output:0'N_of_Paper/strided_slice/stack:output:0)N_of_Paper/strided_slice/stack_1:output:0)N_of_Paper/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask\
N_of_Paper/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
N_of_Paper/Reshape/shapePack!N_of_Paper/strided_slice:output:0#N_of_Paper/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
N_of_Paper/ReshapeReshapefeatures_n_of_paper!N_of_Paper/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????O
N_of_SCI/ShapeShapefeatures_n_of_sci*
T0*
_output_shapes
:f
N_of_SCI/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: h
N_of_SCI/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:h
N_of_SCI/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
N_of_SCI/strided_sliceStridedSliceN_of_SCI/Shape:output:0%N_of_SCI/strided_slice/stack:output:0'N_of_SCI/strided_slice/stack_1:output:0'N_of_SCI/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskZ
N_of_SCI/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
N_of_SCI/Reshape/shapePackN_of_SCI/strided_slice:output:0!N_of_SCI/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
N_of_SCI/ReshapeReshapefeatures_n_of_sciN_of_SCI/Reshape/shape:output:0*
T0*'
_output_shapes
:??????????
<National_Strategy_2_indicator/to_sparse_input/ignore_value/xConst*
_output_shapes
: *
dtype0*
valueB :
??????????
:National_Strategy_2_indicator/to_sparse_input/ignore_valueCastENational_Strategy_2_indicator/to_sparse_input/ignore_value/x:output:0*

DstT0	*

SrcT0*
_output_shapes
: ?
6National_Strategy_2_indicator/to_sparse_input/NotEqualNotEqualfeatures_national_strategy_2>National_Strategy_2_indicator/to_sparse_input/ignore_value:y:0*
T0	*'
_output_shapes
:??????????
5National_Strategy_2_indicator/to_sparse_input/indicesWhere:National_Strategy_2_indicator/to_sparse_input/NotEqual:z:0*'
_output_shapes
:??????????
4National_Strategy_2_indicator/to_sparse_input/valuesGatherNdfeatures_national_strategy_2=National_Strategy_2_indicator/to_sparse_input/indices:index:0*
Tindices0	*
Tparams0	*#
_output_shapes
:??????????
9National_Strategy_2_indicator/to_sparse_input/dense_shapeShapefeatures_national_strategy_2*
T0	*
_output_shapes
:*
out_type0	?
;National_Strategy_2_indicator/None_Lookup/LookupTableFindV2LookupTableFindV2Hnational_strategy_2_indicator_none_lookup_lookuptablefindv2_table_handle=National_Strategy_2_indicator/to_sparse_input/values:output:0Inational_strategy_2_indicator_none_lookup_lookuptablefindv2_default_value*	
Tin0	*

Tout0	*#
_output_shapes
:??????????
9National_Strategy_2_indicator/SparseToDense/default_valueConst*
_output_shapes
: *
dtype0	*
valueB	 R
??????????
+National_Strategy_2_indicator/SparseToDenseSparseToDense=National_Strategy_2_indicator/to_sparse_input/indices:index:0BNational_Strategy_2_indicator/to_sparse_input/dense_shape:output:0DNational_Strategy_2_indicator/None_Lookup/LookupTableFindV2:values:0BNational_Strategy_2_indicator/SparseToDense/default_value:output:0*
T0	*
Tindices0	*'
_output_shapes
:?????????p
+National_Strategy_2_indicator/one_hot/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??r
-National_Strategy_2_indicator/one_hot/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    m
+National_Strategy_2_indicator/one_hot/depthConst*
_output_shapes
: *
dtype0*
value	B :?
%National_Strategy_2_indicator/one_hotOneHot3National_Strategy_2_indicator/SparseToDense:dense:04National_Strategy_2_indicator/one_hot/depth:output:04National_Strategy_2_indicator/one_hot/Const:output:06National_Strategy_2_indicator/one_hot/Const_1:output:0*
T0*+
_output_shapes
:??????????
3National_Strategy_2_indicator/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
??????????
!National_Strategy_2_indicator/SumSum.National_Strategy_2_indicator/one_hot:output:0<National_Strategy_2_indicator/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:?????????}
#National_Strategy_2_indicator/ShapeShape*National_Strategy_2_indicator/Sum:output:0*
T0*
_output_shapes
:{
1National_Strategy_2_indicator/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: }
3National_Strategy_2_indicator/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:}
3National_Strategy_2_indicator/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
+National_Strategy_2_indicator/strided_sliceStridedSlice,National_Strategy_2_indicator/Shape:output:0:National_Strategy_2_indicator/strided_slice/stack:output:0<National_Strategy_2_indicator/strided_slice/stack_1:output:0<National_Strategy_2_indicator/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_masko
-National_Strategy_2_indicator/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
+National_Strategy_2_indicator/Reshape/shapePack4National_Strategy_2_indicator/strided_slice:output:06National_Strategy_2_indicator/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
%National_Strategy_2_indicator/ReshapeReshape*National_Strategy_2_indicator/Sum:output:04National_Strategy_2_indicator/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????{
0RnD_Org_indicator/to_sparse_input/ignore_value/xConst*
_output_shapes
: *
dtype0*
valueB :
??????????
.RnD_Org_indicator/to_sparse_input/ignore_valueCast9RnD_Org_indicator/to_sparse_input/ignore_value/x:output:0*

DstT0	*

SrcT0*
_output_shapes
: ?
*RnD_Org_indicator/to_sparse_input/NotEqualNotEqualfeatures_rnd_org2RnD_Org_indicator/to_sparse_input/ignore_value:y:0*
T0	*'
_output_shapes
:??????????
)RnD_Org_indicator/to_sparse_input/indicesWhere.RnD_Org_indicator/to_sparse_input/NotEqual:z:0*'
_output_shapes
:??????????
(RnD_Org_indicator/to_sparse_input/valuesGatherNdfeatures_rnd_org1RnD_Org_indicator/to_sparse_input/indices:index:0*
Tindices0	*
Tparams0	*#
_output_shapes
:?????????}
-RnD_Org_indicator/to_sparse_input/dense_shapeShapefeatures_rnd_org*
T0	*
_output_shapes
:*
out_type0	?
/RnD_Org_indicator/None_Lookup/LookupTableFindV2LookupTableFindV2<rnd_org_indicator_none_lookup_lookuptablefindv2_table_handle1RnD_Org_indicator/to_sparse_input/values:output:0=rnd_org_indicator_none_lookup_lookuptablefindv2_default_value*	
Tin0	*

Tout0	*#
_output_shapes
:?????????x
-RnD_Org_indicator/SparseToDense/default_valueConst*
_output_shapes
: *
dtype0	*
valueB	 R
??????????
RnD_Org_indicator/SparseToDenseSparseToDense1RnD_Org_indicator/to_sparse_input/indices:index:06RnD_Org_indicator/to_sparse_input/dense_shape:output:08RnD_Org_indicator/None_Lookup/LookupTableFindV2:values:06RnD_Org_indicator/SparseToDense/default_value:output:0*
T0	*
Tindices0	*'
_output_shapes
:?????????d
RnD_Org_indicator/one_hot/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??f
!RnD_Org_indicator/one_hot/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    a
RnD_Org_indicator/one_hot/depthConst*
_output_shapes
: *
dtype0*
value	B :?
RnD_Org_indicator/one_hotOneHot'RnD_Org_indicator/SparseToDense:dense:0(RnD_Org_indicator/one_hot/depth:output:0(RnD_Org_indicator/one_hot/Const:output:0*RnD_Org_indicator/one_hot/Const_1:output:0*
T0*+
_output_shapes
:?????????z
'RnD_Org_indicator/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
??????????
RnD_Org_indicator/SumSum"RnD_Org_indicator/one_hot:output:00RnD_Org_indicator/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:?????????e
RnD_Org_indicator/ShapeShapeRnD_Org_indicator/Sum:output:0*
T0*
_output_shapes
:o
%RnD_Org_indicator/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: q
'RnD_Org_indicator/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:q
'RnD_Org_indicator/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
RnD_Org_indicator/strided_sliceStridedSlice RnD_Org_indicator/Shape:output:0.RnD_Org_indicator/strided_slice/stack:output:00RnD_Org_indicator/strided_slice/stack_1:output:00RnD_Org_indicator/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskc
!RnD_Org_indicator/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
RnD_Org_indicator/Reshape/shapePack(RnD_Org_indicator/strided_slice:output:0*RnD_Org_indicator/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
RnD_Org_indicator/ReshapeReshapeRnD_Org_indicator/Sum:output:0(RnD_Org_indicator/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????}
2RnD_Stage_indicator/to_sparse_input/ignore_value/xConst*
_output_shapes
: *
dtype0*
valueB :
??????????
0RnD_Stage_indicator/to_sparse_input/ignore_valueCast;RnD_Stage_indicator/to_sparse_input/ignore_value/x:output:0*

DstT0	*

SrcT0*
_output_shapes
: ?
,RnD_Stage_indicator/to_sparse_input/NotEqualNotEqualfeatures_rnd_stage4RnD_Stage_indicator/to_sparse_input/ignore_value:y:0*
T0	*'
_output_shapes
:??????????
+RnD_Stage_indicator/to_sparse_input/indicesWhere0RnD_Stage_indicator/to_sparse_input/NotEqual:z:0*'
_output_shapes
:??????????
*RnD_Stage_indicator/to_sparse_input/valuesGatherNdfeatures_rnd_stage3RnD_Stage_indicator/to_sparse_input/indices:index:0*
Tindices0	*
Tparams0	*#
_output_shapes
:??????????
/RnD_Stage_indicator/to_sparse_input/dense_shapeShapefeatures_rnd_stage*
T0	*
_output_shapes
:*
out_type0	?
1RnD_Stage_indicator/None_Lookup/LookupTableFindV2LookupTableFindV2>rnd_stage_indicator_none_lookup_lookuptablefindv2_table_handle3RnD_Stage_indicator/to_sparse_input/values:output:0?rnd_stage_indicator_none_lookup_lookuptablefindv2_default_value*	
Tin0	*

Tout0	*#
_output_shapes
:?????????z
/RnD_Stage_indicator/SparseToDense/default_valueConst*
_output_shapes
: *
dtype0	*
valueB	 R
??????????
!RnD_Stage_indicator/SparseToDenseSparseToDense3RnD_Stage_indicator/to_sparse_input/indices:index:08RnD_Stage_indicator/to_sparse_input/dense_shape:output:0:RnD_Stage_indicator/None_Lookup/LookupTableFindV2:values:08RnD_Stage_indicator/SparseToDense/default_value:output:0*
T0	*
Tindices0	*'
_output_shapes
:?????????f
!RnD_Stage_indicator/one_hot/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??h
#RnD_Stage_indicator/one_hot/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    c
!RnD_Stage_indicator/one_hot/depthConst*
_output_shapes
: *
dtype0*
value	B :?
RnD_Stage_indicator/one_hotOneHot)RnD_Stage_indicator/SparseToDense:dense:0*RnD_Stage_indicator/one_hot/depth:output:0*RnD_Stage_indicator/one_hot/Const:output:0,RnD_Stage_indicator/one_hot/Const_1:output:0*
T0*+
_output_shapes
:?????????|
)RnD_Stage_indicator/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
??????????
RnD_Stage_indicator/SumSum$RnD_Stage_indicator/one_hot:output:02RnD_Stage_indicator/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:?????????i
RnD_Stage_indicator/ShapeShape RnD_Stage_indicator/Sum:output:0*
T0*
_output_shapes
:q
'RnD_Stage_indicator/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: s
)RnD_Stage_indicator/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:s
)RnD_Stage_indicator/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
!RnD_Stage_indicator/strided_sliceStridedSlice"RnD_Stage_indicator/Shape:output:00RnD_Stage_indicator/strided_slice/stack:output:02RnD_Stage_indicator/strided_slice/stack_1:output:02RnD_Stage_indicator/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maske
#RnD_Stage_indicator/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
!RnD_Stage_indicator/Reshape/shapePack*RnD_Stage_indicator/strided_slice:output:0,RnD_Stage_indicator/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
RnD_Stage_indicator/ReshapeReshape RnD_Stage_indicator/Sum:output:0*RnD_Stage_indicator/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????u
4STP_Code_11_indicator/to_sparse_input/ignore_value/xConst*
_output_shapes
: *
dtype0*
valueB B ?
.STP_Code_11_indicator/to_sparse_input/NotEqualNotEqualfeatures_stp_code_11=STP_Code_11_indicator/to_sparse_input/ignore_value/x:output:0*
T0*'
_output_shapes
:??????????
-STP_Code_11_indicator/to_sparse_input/indicesWhere2STP_Code_11_indicator/to_sparse_input/NotEqual:z:0*'
_output_shapes
:??????????
,STP_Code_11_indicator/to_sparse_input/valuesGatherNdfeatures_stp_code_115STP_Code_11_indicator/to_sparse_input/indices:index:0*
Tindices0	*
Tparams0*#
_output_shapes
:??????????
1STP_Code_11_indicator/to_sparse_input/dense_shapeShapefeatures_stp_code_11*
T0*
_output_shapes
:*
out_type0	?
3STP_Code_11_indicator/None_Lookup/LookupTableFindV2LookupTableFindV2@stp_code_11_indicator_none_lookup_lookuptablefindv2_table_handle5STP_Code_11_indicator/to_sparse_input/values:output:0Astp_code_11_indicator_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:?????????|
1STP_Code_11_indicator/SparseToDense/default_valueConst*
_output_shapes
: *
dtype0	*
valueB	 R
??????????
#STP_Code_11_indicator/SparseToDenseSparseToDense5STP_Code_11_indicator/to_sparse_input/indices:index:0:STP_Code_11_indicator/to_sparse_input/dense_shape:output:0<STP_Code_11_indicator/None_Lookup/LookupTableFindV2:values:0:STP_Code_11_indicator/SparseToDense/default_value:output:0*
T0	*
Tindices0	*'
_output_shapes
:?????????h
#STP_Code_11_indicator/one_hot/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??j
%STP_Code_11_indicator/one_hot/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    f
#STP_Code_11_indicator/one_hot/depthConst*
_output_shapes
: *
dtype0*
value
B :??
STP_Code_11_indicator/one_hotOneHot+STP_Code_11_indicator/SparseToDense:dense:0,STP_Code_11_indicator/one_hot/depth:output:0,STP_Code_11_indicator/one_hot/Const:output:0.STP_Code_11_indicator/one_hot/Const_1:output:0*
T0*,
_output_shapes
:??????????~
+STP_Code_11_indicator/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
??????????
STP_Code_11_indicator/SumSum&STP_Code_11_indicator/one_hot:output:04STP_Code_11_indicator/Sum/reduction_indices:output:0*
T0*(
_output_shapes
:??????????m
STP_Code_11_indicator/ShapeShape"STP_Code_11_indicator/Sum:output:0*
T0*
_output_shapes
:s
)STP_Code_11_indicator/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: u
+STP_Code_11_indicator/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+STP_Code_11_indicator/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
#STP_Code_11_indicator/strided_sliceStridedSlice$STP_Code_11_indicator/Shape:output:02STP_Code_11_indicator/strided_slice/stack:output:04STP_Code_11_indicator/strided_slice/stack_1:output:04STP_Code_11_indicator/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskh
%STP_Code_11_indicator/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value
B :??
#STP_Code_11_indicator/Reshape/shapePack,STP_Code_11_indicator/strided_slice:output:0.STP_Code_11_indicator/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
STP_Code_11_indicator/ReshapeReshape"STP_Code_11_indicator/Sum:output:0,STP_Code_11_indicator/Reshape/shape:output:0*
T0*(
_output_shapes
:??????????a
STP_Code_1_Weight/ShapeShapefeatures_stp_code_1_weight*
T0*
_output_shapes
:o
%STP_Code_1_Weight/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: q
'STP_Code_1_Weight/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:q
'STP_Code_1_Weight/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
STP_Code_1_Weight/strided_sliceStridedSlice STP_Code_1_Weight/Shape:output:0.STP_Code_1_Weight/strided_slice/stack:output:00STP_Code_1_Weight/strided_slice/stack_1:output:00STP_Code_1_Weight/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskc
!STP_Code_1_Weight/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
STP_Code_1_Weight/Reshape/shapePack(STP_Code_1_Weight/strided_slice:output:0*STP_Code_1_Weight/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
STP_Code_1_Weight/ReshapeReshapefeatures_stp_code_1_weight(STP_Code_1_Weight/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????u
4STP_Code_21_indicator/to_sparse_input/ignore_value/xConst*
_output_shapes
: *
dtype0*
valueB B ?
.STP_Code_21_indicator/to_sparse_input/NotEqualNotEqualfeatures_stp_code_21=STP_Code_21_indicator/to_sparse_input/ignore_value/x:output:0*
T0*'
_output_shapes
:??????????
-STP_Code_21_indicator/to_sparse_input/indicesWhere2STP_Code_21_indicator/to_sparse_input/NotEqual:z:0*'
_output_shapes
:??????????
,STP_Code_21_indicator/to_sparse_input/valuesGatherNdfeatures_stp_code_215STP_Code_21_indicator/to_sparse_input/indices:index:0*
Tindices0	*
Tparams0*#
_output_shapes
:??????????
1STP_Code_21_indicator/to_sparse_input/dense_shapeShapefeatures_stp_code_21*
T0*
_output_shapes
:*
out_type0	?
3STP_Code_21_indicator/None_Lookup/LookupTableFindV2LookupTableFindV2@stp_code_21_indicator_none_lookup_lookuptablefindv2_table_handle5STP_Code_21_indicator/to_sparse_input/values:output:0Astp_code_21_indicator_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:?????????|
1STP_Code_21_indicator/SparseToDense/default_valueConst*
_output_shapes
: *
dtype0	*
valueB	 R
??????????
#STP_Code_21_indicator/SparseToDenseSparseToDense5STP_Code_21_indicator/to_sparse_input/indices:index:0:STP_Code_21_indicator/to_sparse_input/dense_shape:output:0<STP_Code_21_indicator/None_Lookup/LookupTableFindV2:values:0:STP_Code_21_indicator/SparseToDense/default_value:output:0*
T0	*
Tindices0	*'
_output_shapes
:?????????h
#STP_Code_21_indicator/one_hot/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??j
%STP_Code_21_indicator/one_hot/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    f
#STP_Code_21_indicator/one_hot/depthConst*
_output_shapes
: *
dtype0*
value
B :??
STP_Code_21_indicator/one_hotOneHot+STP_Code_21_indicator/SparseToDense:dense:0,STP_Code_21_indicator/one_hot/depth:output:0,STP_Code_21_indicator/one_hot/Const:output:0.STP_Code_21_indicator/one_hot/Const_1:output:0*
T0*,
_output_shapes
:??????????~
+STP_Code_21_indicator/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
??????????
STP_Code_21_indicator/SumSum&STP_Code_21_indicator/one_hot:output:04STP_Code_21_indicator/Sum/reduction_indices:output:0*
T0*(
_output_shapes
:??????????m
STP_Code_21_indicator/ShapeShape"STP_Code_21_indicator/Sum:output:0*
T0*
_output_shapes
:s
)STP_Code_21_indicator/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: u
+STP_Code_21_indicator/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+STP_Code_21_indicator/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
#STP_Code_21_indicator/strided_sliceStridedSlice$STP_Code_21_indicator/Shape:output:02STP_Code_21_indicator/strided_slice/stack:output:04STP_Code_21_indicator/strided_slice/stack_1:output:04STP_Code_21_indicator/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskh
%STP_Code_21_indicator/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value
B :??
#STP_Code_21_indicator/Reshape/shapePack,STP_Code_21_indicator/strided_slice:output:0.STP_Code_21_indicator/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
STP_Code_21_indicator/ReshapeReshape"STP_Code_21_indicator/Sum:output:0,STP_Code_21_indicator/Reshape/shape:output:0*
T0*(
_output_shapes
:??????????a
STP_Code_2_Weight/ShapeShapefeatures_stp_code_2_weight*
T0*
_output_shapes
:o
%STP_Code_2_Weight/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: q
'STP_Code_2_Weight/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:q
'STP_Code_2_Weight/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
STP_Code_2_Weight/strided_sliceStridedSlice STP_Code_2_Weight/Shape:output:0.STP_Code_2_Weight/strided_slice/stack:output:00STP_Code_2_Weight/strided_slice/stack_1:output:00STP_Code_2_Weight/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskc
!STP_Code_2_Weight/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
STP_Code_2_Weight/Reshape/shapePack(STP_Code_2_Weight/strided_slice:output:0*STP_Code_2_Weight/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
STP_Code_2_Weight/ReshapeReshapefeatures_stp_code_2_weight(STP_Code_2_Weight/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????z
/SixT_2_indicator/to_sparse_input/ignore_value/xConst*
_output_shapes
: *
dtype0*
valueB :
??????????
-SixT_2_indicator/to_sparse_input/ignore_valueCast8SixT_2_indicator/to_sparse_input/ignore_value/x:output:0*

DstT0	*

SrcT0*
_output_shapes
: ?
)SixT_2_indicator/to_sparse_input/NotEqualNotEqualfeatures_sixt_21SixT_2_indicator/to_sparse_input/ignore_value:y:0*
T0	*'
_output_shapes
:??????????
(SixT_2_indicator/to_sparse_input/indicesWhere-SixT_2_indicator/to_sparse_input/NotEqual:z:0*'
_output_shapes
:??????????
'SixT_2_indicator/to_sparse_input/valuesGatherNdfeatures_sixt_20SixT_2_indicator/to_sparse_input/indices:index:0*
Tindices0	*
Tparams0	*#
_output_shapes
:?????????{
,SixT_2_indicator/to_sparse_input/dense_shapeShapefeatures_sixt_2*
T0	*
_output_shapes
:*
out_type0	?
.SixT_2_indicator/None_Lookup/LookupTableFindV2LookupTableFindV2;sixt_2_indicator_none_lookup_lookuptablefindv2_table_handle0SixT_2_indicator/to_sparse_input/values:output:0<sixt_2_indicator_none_lookup_lookuptablefindv2_default_value*	
Tin0	*

Tout0	*#
_output_shapes
:?????????w
,SixT_2_indicator/SparseToDense/default_valueConst*
_output_shapes
: *
dtype0	*
valueB	 R
??????????
SixT_2_indicator/SparseToDenseSparseToDense0SixT_2_indicator/to_sparse_input/indices:index:05SixT_2_indicator/to_sparse_input/dense_shape:output:07SixT_2_indicator/None_Lookup/LookupTableFindV2:values:05SixT_2_indicator/SparseToDense/default_value:output:0*
T0	*
Tindices0	*'
_output_shapes
:?????????c
SixT_2_indicator/one_hot/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??e
 SixT_2_indicator/one_hot/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    `
SixT_2_indicator/one_hot/depthConst*
_output_shapes
: *
dtype0*
value	B :?
SixT_2_indicator/one_hotOneHot&SixT_2_indicator/SparseToDense:dense:0'SixT_2_indicator/one_hot/depth:output:0'SixT_2_indicator/one_hot/Const:output:0)SixT_2_indicator/one_hot/Const_1:output:0*
T0*+
_output_shapes
:?????????y
&SixT_2_indicator/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
??????????
SixT_2_indicator/SumSum!SixT_2_indicator/one_hot:output:0/SixT_2_indicator/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:?????????c
SixT_2_indicator/ShapeShapeSixT_2_indicator/Sum:output:0*
T0*
_output_shapes
:n
$SixT_2_indicator/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: p
&SixT_2_indicator/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:p
&SixT_2_indicator/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
SixT_2_indicator/strided_sliceStridedSliceSixT_2_indicator/Shape:output:0-SixT_2_indicator/strided_slice/stack:output:0/SixT_2_indicator/strided_slice/stack_1:output:0/SixT_2_indicator/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskb
 SixT_2_indicator/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
SixT_2_indicator/Reshape/shapePack'SixT_2_indicator/strided_slice:output:0)SixT_2_indicator/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
SixT_2_indicator/ReshapeReshapeSixT_2_indicator/Sum:output:0'SixT_2_indicator/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????G

Year/ShapeShapefeatures_year*
T0*
_output_shapes
:b
Year/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: d
Year/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:d
Year/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
Year/strided_sliceStridedSliceYear/Shape:output:0!Year/strided_slice/stack:output:0#Year/strided_slice/stack_1:output:0#Year/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskV
Year/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
Year/Reshape/shapePackYear/strided_slice:output:0Year/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:u
Year/ReshapeReshapefeatures_yearYear/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????V
concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
??????????
concatConcatV2*Application_Area_1_Weight/Reshape:output:0-Application_Area_1_indicator/Reshape:output:0*Application_Area_2_Weight/Reshape:output:0-Application_Area_2_indicator/Reshape:output:0(Cowork_Abroad_indicator/Reshape:output:0%Cowork_Cor_indicator/Reshape:output:0&Cowork_Inst_indicator/Reshape:output:0%Cowork_Uni_indicator/Reshape:output:0%Cowork_etc_indicator/Reshape:output:0&Econ_Social_indicator/Reshape:output:0%Green_Tech_indicator/Reshape:output:0Log_Duration/Reshape:output:0Log_RnD_Fund/Reshape:output:0%Multi_Year_indicator/Reshape:output:0N_Patent_App/Reshape:output:0N_Patent_Reg/Reshape:output:0#N_of_Korean_Patent/Reshape:output:0N_of_Paper/Reshape:output:0N_of_SCI/Reshape:output:0.National_Strategy_2_indicator/Reshape:output:0"RnD_Org_indicator/Reshape:output:0$RnD_Stage_indicator/Reshape:output:0&STP_Code_11_indicator/Reshape:output:0"STP_Code_1_Weight/Reshape:output:0&STP_Code_21_indicator/Reshape:output:0"STP_Code_2_Weight/Reshape:output:0!SixT_2_indicator/Reshape:output:0Year/Reshape:output:0concat/axis:output:0*
N*
T0*(
_output_shapes
:??????????_
IdentityIdentityconcat:output:0^NoOp*
T0*(
_output_shapes
:???????????
NoOpNoOp;^Application_Area_1_indicator/None_Lookup/LookupTableFindV2;^Application_Area_2_indicator/None_Lookup/LookupTableFindV26^Cowork_Abroad_indicator/None_Lookup/LookupTableFindV23^Cowork_Cor_indicator/None_Lookup/LookupTableFindV24^Cowork_Inst_indicator/None_Lookup/LookupTableFindV23^Cowork_Uni_indicator/None_Lookup/LookupTableFindV23^Cowork_etc_indicator/None_Lookup/LookupTableFindV24^Econ_Social_indicator/None_Lookup/LookupTableFindV23^Green_Tech_indicator/None_Lookup/LookupTableFindV23^Multi_Year_indicator/None_Lookup/LookupTableFindV2<^National_Strategy_2_indicator/None_Lookup/LookupTableFindV20^RnD_Org_indicator/None_Lookup/LookupTableFindV22^RnD_Stage_indicator/None_Lookup/LookupTableFindV24^STP_Code_11_indicator/None_Lookup/LookupTableFindV24^STP_Code_21_indicator/None_Lookup/LookupTableFindV2/^SixT_2_indicator/None_Lookup/LookupTableFindV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2x
:Application_Area_1_indicator/None_Lookup/LookupTableFindV2:Application_Area_1_indicator/None_Lookup/LookupTableFindV22x
:Application_Area_2_indicator/None_Lookup/LookupTableFindV2:Application_Area_2_indicator/None_Lookup/LookupTableFindV22n
5Cowork_Abroad_indicator/None_Lookup/LookupTableFindV25Cowork_Abroad_indicator/None_Lookup/LookupTableFindV22h
2Cowork_Cor_indicator/None_Lookup/LookupTableFindV22Cowork_Cor_indicator/None_Lookup/LookupTableFindV22j
3Cowork_Inst_indicator/None_Lookup/LookupTableFindV23Cowork_Inst_indicator/None_Lookup/LookupTableFindV22h
2Cowork_Uni_indicator/None_Lookup/LookupTableFindV22Cowork_Uni_indicator/None_Lookup/LookupTableFindV22h
2Cowork_etc_indicator/None_Lookup/LookupTableFindV22Cowork_etc_indicator/None_Lookup/LookupTableFindV22j
3Econ_Social_indicator/None_Lookup/LookupTableFindV23Econ_Social_indicator/None_Lookup/LookupTableFindV22h
2Green_Tech_indicator/None_Lookup/LookupTableFindV22Green_Tech_indicator/None_Lookup/LookupTableFindV22h
2Multi_Year_indicator/None_Lookup/LookupTableFindV22Multi_Year_indicator/None_Lookup/LookupTableFindV22z
;National_Strategy_2_indicator/None_Lookup/LookupTableFindV2;National_Strategy_2_indicator/None_Lookup/LookupTableFindV22b
/RnD_Org_indicator/None_Lookup/LookupTableFindV2/RnD_Org_indicator/None_Lookup/LookupTableFindV22f
1RnD_Stage_indicator/None_Lookup/LookupTableFindV21RnD_Stage_indicator/None_Lookup/LookupTableFindV22j
3STP_Code_11_indicator/None_Lookup/LookupTableFindV23STP_Code_11_indicator/None_Lookup/LookupTableFindV22j
3STP_Code_21_indicator/None_Lookup/LookupTableFindV23STP_Code_21_indicator/None_Lookup/LookupTableFindV22`
.SixT_2_indicator/None_Lookup/LookupTableFindV2.SixT_2_indicator/None_Lookup/LookupTableFindV2:d `
'
_output_shapes
:?????????
5
_user_specified_namefeatures/Application_Area_1:kg
'
_output_shapes
:?????????
<
_user_specified_name$"features/Application_Area_1_Weight:d`
'
_output_shapes
:?????????
5
_user_specified_namefeatures/Application_Area_2:kg
'
_output_shapes
:?????????
<
_user_specified_name$"features/Application_Area_2_Weight:_[
'
_output_shapes
:?????????
0
_user_specified_namefeatures/Cowork_Abroad:\X
'
_output_shapes
:?????????
-
_user_specified_namefeatures/Cowork_Cor:]Y
'
_output_shapes
:?????????
.
_user_specified_namefeatures/Cowork_Inst:\X
'
_output_shapes
:?????????
-
_user_specified_namefeatures/Cowork_Uni:\X
'
_output_shapes
:?????????
-
_user_specified_namefeatures/Cowork_etc:]	Y
'
_output_shapes
:?????????
.
_user_specified_namefeatures/Econ_Social:\
X
'
_output_shapes
:?????????
-
_user_specified_namefeatures/Green_Tech:^Z
'
_output_shapes
:?????????
/
_user_specified_namefeatures/Log_Duration:^Z
'
_output_shapes
:?????????
/
_user_specified_namefeatures/Log_RnD_Fund:\X
'
_output_shapes
:?????????
-
_user_specified_namefeatures/Multi_Year:^Z
'
_output_shapes
:?????????
/
_user_specified_namefeatures/N_Patent_App:^Z
'
_output_shapes
:?????????
/
_user_specified_namefeatures/N_Patent_Reg:d`
'
_output_shapes
:?????????
5
_user_specified_namefeatures/N_of_Korean_Patent:\X
'
_output_shapes
:?????????
-
_user_specified_namefeatures/N_of_Paper:ZV
'
_output_shapes
:?????????
+
_user_specified_namefeatures/N_of_SCI:ea
'
_output_shapes
:?????????
6
_user_specified_namefeatures/National_Strategy_2:YU
'
_output_shapes
:?????????
*
_user_specified_namefeatures/RnD_Org:[W
'
_output_shapes
:?????????
,
_user_specified_namefeatures/RnD_Stage:]Y
'
_output_shapes
:?????????
.
_user_specified_namefeatures/STP_Code_11:c_
'
_output_shapes
:?????????
4
_user_specified_namefeatures/STP_Code_1_Weight:]Y
'
_output_shapes
:?????????
.
_user_specified_namefeatures/STP_Code_21:c_
'
_output_shapes
:?????????
4
_user_specified_namefeatures/STP_Code_2_Weight:XT
'
_output_shapes
:?????????
)
_user_specified_namefeatures/SixT_2:VR
'
_output_shapes
:?????????
'
_user_specified_namefeatures/Year:

_output_shapes
: :

_output_shapes
: :!

_output_shapes
: :#

_output_shapes
: :%

_output_shapes
: :'

_output_shapes
: :)

_output_shapes
: :+

_output_shapes
: :-

_output_shapes
: :/

_output_shapes
: :1

_output_shapes
: :3

_output_shapes
: :5

_output_shapes
: :7

_output_shapes
: :9

_output_shapes
: :;

_output_shapes
: 
?
;
__inference__creator_133835
identity??
hash_tablek

hash_tableHashTableV2*
_output_shapes
: *
	key_dtype0	*
shared_name712*
value_dtype0	W
IdentityIdentityhash_table:table_handle:0^NoOp*
T0*
_output_shapes
: S
NoOpNoOp^hash_table*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 2

hash_table
hash_table
?
?
__inference_<lambda>_1340502
.table_init817_lookuptableimportv2_table_handle*
&table_init817_lookuptableimportv2_keys,
(table_init817_lookuptableimportv2_values	
identity??!table_init817/LookupTableImportV2?
!table_init817/LookupTableImportV2LookupTableImportV2.table_init817_lookuptableimportv2_table_handle&table_init817_lookuptableimportv2_keys(table_init817_lookuptableimportv2_values*	
Tin0*

Tout0	*
_output_shapes
 J
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??L
IdentityIdentityConst:output:0^NoOp*
T0*
_output_shapes
: j
NoOpNoOp"^table_init817/LookupTableImportV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*#
_input_shapes
: :?:?2F
!table_init817/LookupTableImportV2!table_init817/LookupTableImportV2:!

_output_shapes	
:?:!

_output_shapes	
:?
?

?
C__inference_dense_2_layer_call_and_return_conditional_losses_129161

inputs2
matmul_readvariableop_resource:
??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????Q
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:??????????b
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:??????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
(__inference_dense_2_layer_call_fn_133592

inputs
unknown:
??
	unknown_0:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_dense_2_layer_call_and_return_conditional_losses_129161p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:??????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
__inference__initializer_1337352
.table_init443_lookuptableimportv2_table_handle*
&table_init443_lookuptableimportv2_keys,
(table_init443_lookuptableimportv2_values	
identity??!table_init443/LookupTableImportV2?
!table_init443/LookupTableImportV2LookupTableImportV2.table_init443_lookuptableimportv2_table_handle&table_init443_lookuptableimportv2_keys(table_init443_lookuptableimportv2_values*	
Tin0*

Tout0	*
_output_shapes
 G
ConstConst*
_output_shapes
: *
dtype0*
value	B :L
IdentityIdentityConst:output:0^NoOp*
T0*
_output_shapes
: j
NoOpNoOp"^table_init443/LookupTableImportV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*!
_input_shapes
: ::2F
!table_init443/LookupTableImportV2!table_init443/LookupTableImportV2: 

_output_shapes
:: 

_output_shapes
:
?
?
__inference_<lambda>_1339942
.table_init511_lookuptableimportv2_table_handle*
&table_init511_lookuptableimportv2_keys,
(table_init511_lookuptableimportv2_values	
identity??!table_init511/LookupTableImportV2?
!table_init511/LookupTableImportV2LookupTableImportV2.table_init511_lookuptableimportv2_table_handle&table_init511_lookuptableimportv2_keys(table_init511_lookuptableimportv2_values*	
Tin0*

Tout0	*
_output_shapes
 J
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??L
IdentityIdentityConst:output:0^NoOp*
T0*
_output_shapes
: j
NoOpNoOp"^table_init511/LookupTableImportV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*!
_input_shapes
: ::2F
!table_init511/LookupTableImportV2!table_init511/LookupTableImportV2: 

_output_shapes
:: 

_output_shapes
:
?

?
C__inference_dense_3_layer_call_and_return_conditional_losses_129185

inputs1
matmul_readvariableop_resource:	?-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????V
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:?????????Z
IdentityIdentitySigmoid:y:0^NoOp*
T0*'
_output_shapes
:?????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?6
?

/__inference_dense_features_layer_call_fn_132449
features_application_area_1&
"features_application_area_1_weight
features_application_area_2&
"features_application_area_2_weight
features_cowork_abroad
features_cowork_cor
features_cowork_inst
features_cowork_uni
features_cowork_etc
features_econ_social	
features_green_tech	
features_log_duration
features_log_rnd_fund
features_multi_year	
features_n_patent_app
features_n_patent_reg
features_n_of_korean_patent
features_n_of_paper
features_n_of_sci 
features_national_strategy_2	
features_rnd_org	
features_rnd_stage	
features_stp_code_11
features_stp_code_1_weight
features_stp_code_21
features_stp_code_2_weight
features_sixt_2	
features_year
unknown
	unknown_0	
	unknown_1
	unknown_2	
	unknown_3
	unknown_4	
	unknown_5
	unknown_6	
	unknown_7
	unknown_8	
	unknown_9

unknown_10	

unknown_11

unknown_12	

unknown_13

unknown_14	

unknown_15

unknown_16	

unknown_17

unknown_18	

unknown_19

unknown_20	

unknown_21

unknown_22	

unknown_23

unknown_24	

unknown_25

unknown_26	

unknown_27

unknown_28	

unknown_29

unknown_30	
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallfeatures_application_area_1"features_application_area_1_weightfeatures_application_area_2"features_application_area_2_weightfeatures_cowork_abroadfeatures_cowork_corfeatures_cowork_instfeatures_cowork_unifeatures_cowork_etcfeatures_econ_socialfeatures_green_techfeatures_log_durationfeatures_log_rnd_fundfeatures_multi_yearfeatures_n_patent_appfeatures_n_patent_regfeatures_n_of_korean_patentfeatures_n_of_paperfeatures_n_of_scifeatures_national_strategy_2features_rnd_orgfeatures_rnd_stagefeatures_stp_code_11features_stp_code_1_weightfeatures_stp_code_21features_stp_code_2_weightfeatures_sixt_2features_yearunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30*G
Tin@
>2<																							*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_dense_features_layer_call_and_return_conditional_losses_130030p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:??????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:d `
'
_output_shapes
:?????????
5
_user_specified_namefeatures/Application_Area_1:kg
'
_output_shapes
:?????????
<
_user_specified_name$"features/Application_Area_1_Weight:d`
'
_output_shapes
:?????????
5
_user_specified_namefeatures/Application_Area_2:kg
'
_output_shapes
:?????????
<
_user_specified_name$"features/Application_Area_2_Weight:_[
'
_output_shapes
:?????????
0
_user_specified_namefeatures/Cowork_Abroad:\X
'
_output_shapes
:?????????
-
_user_specified_namefeatures/Cowork_Cor:]Y
'
_output_shapes
:?????????
.
_user_specified_namefeatures/Cowork_Inst:\X
'
_output_shapes
:?????????
-
_user_specified_namefeatures/Cowork_Uni:\X
'
_output_shapes
:?????????
-
_user_specified_namefeatures/Cowork_etc:]	Y
'
_output_shapes
:?????????
.
_user_specified_namefeatures/Econ_Social:\
X
'
_output_shapes
:?????????
-
_user_specified_namefeatures/Green_Tech:^Z
'
_output_shapes
:?????????
/
_user_specified_namefeatures/Log_Duration:^Z
'
_output_shapes
:?????????
/
_user_specified_namefeatures/Log_RnD_Fund:\X
'
_output_shapes
:?????????
-
_user_specified_namefeatures/Multi_Year:^Z
'
_output_shapes
:?????????
/
_user_specified_namefeatures/N_Patent_App:^Z
'
_output_shapes
:?????????
/
_user_specified_namefeatures/N_Patent_Reg:d`
'
_output_shapes
:?????????
5
_user_specified_namefeatures/N_of_Korean_Patent:\X
'
_output_shapes
:?????????
-
_user_specified_namefeatures/N_of_Paper:ZV
'
_output_shapes
:?????????
+
_user_specified_namefeatures/N_of_SCI:ea
'
_output_shapes
:?????????
6
_user_specified_namefeatures/National_Strategy_2:YU
'
_output_shapes
:?????????
*
_user_specified_namefeatures/RnD_Org:[W
'
_output_shapes
:?????????
,
_user_specified_namefeatures/RnD_Stage:]Y
'
_output_shapes
:?????????
.
_user_specified_namefeatures/STP_Code_11:c_
'
_output_shapes
:?????????
4
_user_specified_namefeatures/STP_Code_1_Weight:]Y
'
_output_shapes
:?????????
.
_user_specified_namefeatures/STP_Code_21:c_
'
_output_shapes
:?????????
4
_user_specified_namefeatures/STP_Code_2_Weight:XT
'
_output_shapes
:?????????
)
_user_specified_namefeatures/SixT_2:VR
'
_output_shapes
:?????????
'
_user_specified_namefeatures/Year:

_output_shapes
: :

_output_shapes
: :!

_output_shapes
: :#

_output_shapes
: :%

_output_shapes
: :'

_output_shapes
: :)

_output_shapes
: :+

_output_shapes
: :-

_output_shapes
: :/

_output_shapes
: :1

_output_shapes
: :3

_output_shapes
: :5

_output_shapes
: :7

_output_shapes
: :9

_output_shapes
: :;

_output_shapes
: 
?
-
__inference__destroyer_133668
identityG
ConstConst*
_output_shapes
: *
dtype0*
value	B :E
IdentityIdentityConst:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 
?
a
C__inference_dropout_layer_call_and_return_conditional_losses_133524

inputs

identity_1O
IdentityIdentityinputs*
T0*(
_output_shapes
:??????????\

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:??????????"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
__inference_<lambda>_1340182
.table_init635_lookuptableimportv2_table_handle*
&table_init635_lookuptableimportv2_keys	,
(table_init635_lookuptableimportv2_values	
identity??!table_init635/LookupTableImportV2?
!table_init635/LookupTableImportV2LookupTableImportV2.table_init635_lookuptableimportv2_table_handle&table_init635_lookuptableimportv2_keys(table_init635_lookuptableimportv2_values*	
Tin0	*

Tout0	*
_output_shapes
 J
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??L
IdentityIdentityConst:output:0^NoOp*
T0*
_output_shapes
: j
NoOpNoOp"^table_init635/LookupTableImportV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*!
_input_shapes
: ::2F
!table_init635/LookupTableImportV2!table_init635/LookupTableImportV2: 

_output_shapes
:: 

_output_shapes
:
?
;
__inference__creator_133925
identity??
hash_tablek

hash_tableHashTableV2*
_output_shapes
: *
	key_dtype0	*
shared_name904*
value_dtype0	W
IdentityIdentityhash_table:table_handle:0^NoOp*
T0*
_output_shapes
: S
NoOpNoOp^hash_table*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 2

hash_table
hash_table
?
?
__inference_<lambda>_1339462
.table_init299_lookuptableimportv2_table_handle*
&table_init299_lookuptableimportv2_keys,
(table_init299_lookuptableimportv2_values	
identity??!table_init299/LookupTableImportV2?
!table_init299/LookupTableImportV2LookupTableImportV2.table_init299_lookuptableimportv2_table_handle&table_init299_lookuptableimportv2_keys(table_init299_lookuptableimportv2_values*	
Tin0*

Tout0	*
_output_shapes
 J
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??L
IdentityIdentityConst:output:0^NoOp*
T0*
_output_shapes
: j
NoOpNoOp"^table_init299/LookupTableImportV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*!
_input_shapes
: :!:!2F
!table_init299/LookupTableImportV2!table_init299/LookupTableImportV2: 

_output_shapes
:!: 

_output_shapes
:!
?
;
__inference__creator_133853
identity??
hash_tablek

hash_tableHashTableV2*
_output_shapes
: *
	key_dtype0	*
shared_name748*
value_dtype0	W
IdentityIdentityhash_table:table_handle:0^NoOp*
T0*
_output_shapes
: S
NoOpNoOp^hash_table*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 2

hash_table
hash_table
?7
?
+__inference_sequential_layer_call_fn_131022
inputs_application_area_1$
 inputs_application_area_1_weight
inputs_application_area_2$
 inputs_application_area_2_weight
inputs_cowork_abroad
inputs_cowork_cor
inputs_cowork_inst
inputs_cowork_uni
inputs_cowork_etc
inputs_econ_social	
inputs_green_tech	
inputs_log_duration
inputs_log_rnd_fund
inputs_multi_year	
inputs_n_patent_app
inputs_n_patent_reg
inputs_n_of_korean_patent
inputs_n_of_paper
inputs_n_of_sci
inputs_national_strategy_2	
inputs_rnd_org	
inputs_rnd_stage	
inputs_stp_code_11
inputs_stp_code_1_weight
inputs_stp_code_21
inputs_stp_code_2_weight
inputs_sixt_2	
inputs_year
unknown
	unknown_0	
	unknown_1
	unknown_2	
	unknown_3
	unknown_4	
	unknown_5
	unknown_6	
	unknown_7
	unknown_8	
	unknown_9

unknown_10	

unknown_11

unknown_12	

unknown_13

unknown_14	

unknown_15

unknown_16	

unknown_17

unknown_18	

unknown_19

unknown_20	

unknown_21

unknown_22	

unknown_23

unknown_24	

unknown_25

unknown_26	

unknown_27

unknown_28	

unknown_29

unknown_30	

unknown_31:
??

unknown_32:	?

unknown_33:
??

unknown_34:	?

unknown_35:
??

unknown_36:	?

unknown_37:	?

unknown_38:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs_application_area_1 inputs_application_area_1_weightinputs_application_area_2 inputs_application_area_2_weightinputs_cowork_abroadinputs_cowork_corinputs_cowork_instinputs_cowork_uniinputs_cowork_etcinputs_econ_socialinputs_green_techinputs_log_durationinputs_log_rnd_fundinputs_multi_yearinputs_n_patent_appinputs_n_patent_reginputs_n_of_korean_patentinputs_n_of_paperinputs_n_of_sciinputs_national_strategy_2inputs_rnd_orginputs_rnd_stageinputs_stp_code_11inputs_stp_code_1_weightinputs_stp_code_21inputs_stp_code_2_weightinputs_sixt_2inputs_yearunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38*O
TinH
F2D																							*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????**
_read_only_resource_inputs

<=>?@ABC*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_sequential_layer_call_and_return_conditional_losses_129192o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:b ^
'
_output_shapes
:?????????
3
_user_specified_nameinputs/Application_Area_1:ie
'
_output_shapes
:?????????
:
_user_specified_name" inputs/Application_Area_1_Weight:b^
'
_output_shapes
:?????????
3
_user_specified_nameinputs/Application_Area_2:ie
'
_output_shapes
:?????????
:
_user_specified_name" inputs/Application_Area_2_Weight:]Y
'
_output_shapes
:?????????
.
_user_specified_nameinputs/Cowork_Abroad:ZV
'
_output_shapes
:?????????
+
_user_specified_nameinputs/Cowork_Cor:[W
'
_output_shapes
:?????????
,
_user_specified_nameinputs/Cowork_Inst:ZV
'
_output_shapes
:?????????
+
_user_specified_nameinputs/Cowork_Uni:ZV
'
_output_shapes
:?????????
+
_user_specified_nameinputs/Cowork_etc:[	W
'
_output_shapes
:?????????
,
_user_specified_nameinputs/Econ_Social:Z
V
'
_output_shapes
:?????????
+
_user_specified_nameinputs/Green_Tech:\X
'
_output_shapes
:?????????
-
_user_specified_nameinputs/Log_Duration:\X
'
_output_shapes
:?????????
-
_user_specified_nameinputs/Log_RnD_Fund:ZV
'
_output_shapes
:?????????
+
_user_specified_nameinputs/Multi_Year:\X
'
_output_shapes
:?????????
-
_user_specified_nameinputs/N_Patent_App:\X
'
_output_shapes
:?????????
-
_user_specified_nameinputs/N_Patent_Reg:b^
'
_output_shapes
:?????????
3
_user_specified_nameinputs/N_of_Korean_Patent:ZV
'
_output_shapes
:?????????
+
_user_specified_nameinputs/N_of_Paper:XT
'
_output_shapes
:?????????
)
_user_specified_nameinputs/N_of_SCI:c_
'
_output_shapes
:?????????
4
_user_specified_nameinputs/National_Strategy_2:WS
'
_output_shapes
:?????????
(
_user_specified_nameinputs/RnD_Org:YU
'
_output_shapes
:?????????
*
_user_specified_nameinputs/RnD_Stage:[W
'
_output_shapes
:?????????
,
_user_specified_nameinputs/STP_Code_11:a]
'
_output_shapes
:?????????
2
_user_specified_nameinputs/STP_Code_1_Weight:[W
'
_output_shapes
:?????????
,
_user_specified_nameinputs/STP_Code_21:a]
'
_output_shapes
:?????????
2
_user_specified_nameinputs/STP_Code_2_Weight:VR
'
_output_shapes
:?????????
'
_user_specified_nameinputs/SixT_2:TP
'
_output_shapes
:?????????
%
_user_specified_nameinputs/Year:

_output_shapes
: :

_output_shapes
: :!

_output_shapes
: :#

_output_shapes
: :%

_output_shapes
: :'

_output_shapes
: :)

_output_shapes
: :+

_output_shapes
: :-

_output_shapes
: :/

_output_shapes
: :1

_output_shapes
: :3

_output_shapes
: :5

_output_shapes
: :7

_output_shapes
: :9

_output_shapes
: :;

_output_shapes
: 
?	
d
E__inference_dropout_2_layer_call_and_return_conditional_losses_129305

inputs
identity?R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??e
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:??????????C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????p
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????j
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:??????????Z
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:??????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
??
?
J__inference_dense_features_layer_call_and_return_conditional_losses_130030
features

features_1

features_2

features_3

features_4

features_5

features_6

features_7

features_8

features_9	
features_10	
features_11
features_12
features_13	
features_14
features_15
features_16
features_17
features_18
features_19	
features_20	
features_21	
features_22
features_23
features_24
features_25
features_26	
features_27K
Gapplication_area_1_indicator_none_lookup_lookuptablefindv2_table_handleL
Happlication_area_1_indicator_none_lookup_lookuptablefindv2_default_value	K
Gapplication_area_2_indicator_none_lookup_lookuptablefindv2_table_handleL
Happlication_area_2_indicator_none_lookup_lookuptablefindv2_default_value	F
Bcowork_abroad_indicator_none_lookup_lookuptablefindv2_table_handleG
Ccowork_abroad_indicator_none_lookup_lookuptablefindv2_default_value	C
?cowork_cor_indicator_none_lookup_lookuptablefindv2_table_handleD
@cowork_cor_indicator_none_lookup_lookuptablefindv2_default_value	D
@cowork_inst_indicator_none_lookup_lookuptablefindv2_table_handleE
Acowork_inst_indicator_none_lookup_lookuptablefindv2_default_value	C
?cowork_uni_indicator_none_lookup_lookuptablefindv2_table_handleD
@cowork_uni_indicator_none_lookup_lookuptablefindv2_default_value	C
?cowork_etc_indicator_none_lookup_lookuptablefindv2_table_handleD
@cowork_etc_indicator_none_lookup_lookuptablefindv2_default_value	D
@econ_social_indicator_none_lookup_lookuptablefindv2_table_handleE
Aecon_social_indicator_none_lookup_lookuptablefindv2_default_value	C
?green_tech_indicator_none_lookup_lookuptablefindv2_table_handleD
@green_tech_indicator_none_lookup_lookuptablefindv2_default_value	C
?multi_year_indicator_none_lookup_lookuptablefindv2_table_handleD
@multi_year_indicator_none_lookup_lookuptablefindv2_default_value	L
Hnational_strategy_2_indicator_none_lookup_lookuptablefindv2_table_handleM
Inational_strategy_2_indicator_none_lookup_lookuptablefindv2_default_value	@
<rnd_org_indicator_none_lookup_lookuptablefindv2_table_handleA
=rnd_org_indicator_none_lookup_lookuptablefindv2_default_value	B
>rnd_stage_indicator_none_lookup_lookuptablefindv2_table_handleC
?rnd_stage_indicator_none_lookup_lookuptablefindv2_default_value	D
@stp_code_11_indicator_none_lookup_lookuptablefindv2_table_handleE
Astp_code_11_indicator_none_lookup_lookuptablefindv2_default_value	D
@stp_code_21_indicator_none_lookup_lookuptablefindv2_table_handleE
Astp_code_21_indicator_none_lookup_lookuptablefindv2_default_value	?
;sixt_2_indicator_none_lookup_lookuptablefindv2_table_handle@
<sixt_2_indicator_none_lookup_lookuptablefindv2_default_value	
identity??:Application_Area_1_indicator/None_Lookup/LookupTableFindV2?:Application_Area_2_indicator/None_Lookup/LookupTableFindV2?5Cowork_Abroad_indicator/None_Lookup/LookupTableFindV2?2Cowork_Cor_indicator/None_Lookup/LookupTableFindV2?3Cowork_Inst_indicator/None_Lookup/LookupTableFindV2?2Cowork_Uni_indicator/None_Lookup/LookupTableFindV2?2Cowork_etc_indicator/None_Lookup/LookupTableFindV2?3Econ_Social_indicator/None_Lookup/LookupTableFindV2?2Green_Tech_indicator/None_Lookup/LookupTableFindV2?2Multi_Year_indicator/None_Lookup/LookupTableFindV2?;National_Strategy_2_indicator/None_Lookup/LookupTableFindV2?/RnD_Org_indicator/None_Lookup/LookupTableFindV2?1RnD_Stage_indicator/None_Lookup/LookupTableFindV2?3STP_Code_11_indicator/None_Lookup/LookupTableFindV2?3STP_Code_21_indicator/None_Lookup/LookupTableFindV2?.SixT_2_indicator/None_Lookup/LookupTableFindV2Y
Application_Area_1_Weight/ShapeShape
features_1*
T0*
_output_shapes
:w
-Application_Area_1_Weight/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: y
/Application_Area_1_Weight/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:y
/Application_Area_1_Weight/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
'Application_Area_1_Weight/strided_sliceStridedSlice(Application_Area_1_Weight/Shape:output:06Application_Area_1_Weight/strided_slice/stack:output:08Application_Area_1_Weight/strided_slice/stack_1:output:08Application_Area_1_Weight/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskk
)Application_Area_1_Weight/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
'Application_Area_1_Weight/Reshape/shapePack0Application_Area_1_Weight/strided_slice:output:02Application_Area_1_Weight/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
!Application_Area_1_Weight/ReshapeReshape
features_10Application_Area_1_Weight/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????|
;Application_Area_1_indicator/to_sparse_input/ignore_value/xConst*
_output_shapes
: *
dtype0*
valueB B ?
5Application_Area_1_indicator/to_sparse_input/NotEqualNotEqualfeaturesDApplication_Area_1_indicator/to_sparse_input/ignore_value/x:output:0*
T0*'
_output_shapes
:??????????
4Application_Area_1_indicator/to_sparse_input/indicesWhere9Application_Area_1_indicator/to_sparse_input/NotEqual:z:0*'
_output_shapes
:??????????
3Application_Area_1_indicator/to_sparse_input/valuesGatherNdfeatures<Application_Area_1_indicator/to_sparse_input/indices:index:0*
Tindices0	*
Tparams0*#
_output_shapes
:??????????
8Application_Area_1_indicator/to_sparse_input/dense_shapeShapefeatures*
T0*
_output_shapes
:*
out_type0	?
:Application_Area_1_indicator/None_Lookup/LookupTableFindV2LookupTableFindV2Gapplication_area_1_indicator_none_lookup_lookuptablefindv2_table_handle<Application_Area_1_indicator/to_sparse_input/values:output:0Happlication_area_1_indicator_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:??????????
8Application_Area_1_indicator/SparseToDense/default_valueConst*
_output_shapes
: *
dtype0	*
valueB	 R
??????????
*Application_Area_1_indicator/SparseToDenseSparseToDense<Application_Area_1_indicator/to_sparse_input/indices:index:0AApplication_Area_1_indicator/to_sparse_input/dense_shape:output:0CApplication_Area_1_indicator/None_Lookup/LookupTableFindV2:values:0AApplication_Area_1_indicator/SparseToDense/default_value:output:0*
T0	*
Tindices0	*'
_output_shapes
:?????????o
*Application_Area_1_indicator/one_hot/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??q
,Application_Area_1_indicator/one_hot/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    l
*Application_Area_1_indicator/one_hot/depthConst*
_output_shapes
: *
dtype0*
value	B :!?
$Application_Area_1_indicator/one_hotOneHot2Application_Area_1_indicator/SparseToDense:dense:03Application_Area_1_indicator/one_hot/depth:output:03Application_Area_1_indicator/one_hot/Const:output:05Application_Area_1_indicator/one_hot/Const_1:output:0*
T0*+
_output_shapes
:?????????!?
2Application_Area_1_indicator/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
??????????
 Application_Area_1_indicator/SumSum-Application_Area_1_indicator/one_hot:output:0;Application_Area_1_indicator/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:?????????!{
"Application_Area_1_indicator/ShapeShape)Application_Area_1_indicator/Sum:output:0*
T0*
_output_shapes
:z
0Application_Area_1_indicator/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: |
2Application_Area_1_indicator/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:|
2Application_Area_1_indicator/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
*Application_Area_1_indicator/strided_sliceStridedSlice+Application_Area_1_indicator/Shape:output:09Application_Area_1_indicator/strided_slice/stack:output:0;Application_Area_1_indicator/strided_slice/stack_1:output:0;Application_Area_1_indicator/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskn
,Application_Area_1_indicator/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :!?
*Application_Area_1_indicator/Reshape/shapePack3Application_Area_1_indicator/strided_slice:output:05Application_Area_1_indicator/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
$Application_Area_1_indicator/ReshapeReshape)Application_Area_1_indicator/Sum:output:03Application_Area_1_indicator/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????!Y
Application_Area_2_Weight/ShapeShape
features_3*
T0*
_output_shapes
:w
-Application_Area_2_Weight/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: y
/Application_Area_2_Weight/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:y
/Application_Area_2_Weight/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
'Application_Area_2_Weight/strided_sliceStridedSlice(Application_Area_2_Weight/Shape:output:06Application_Area_2_Weight/strided_slice/stack:output:08Application_Area_2_Weight/strided_slice/stack_1:output:08Application_Area_2_Weight/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskk
)Application_Area_2_Weight/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
'Application_Area_2_Weight/Reshape/shapePack0Application_Area_2_Weight/strided_slice:output:02Application_Area_2_Weight/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
!Application_Area_2_Weight/ReshapeReshape
features_30Application_Area_2_Weight/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????|
;Application_Area_2_indicator/to_sparse_input/ignore_value/xConst*
_output_shapes
: *
dtype0*
valueB B ?
5Application_Area_2_indicator/to_sparse_input/NotEqualNotEqual
features_2DApplication_Area_2_indicator/to_sparse_input/ignore_value/x:output:0*
T0*'
_output_shapes
:??????????
4Application_Area_2_indicator/to_sparse_input/indicesWhere9Application_Area_2_indicator/to_sparse_input/NotEqual:z:0*'
_output_shapes
:??????????
3Application_Area_2_indicator/to_sparse_input/valuesGatherNd
features_2<Application_Area_2_indicator/to_sparse_input/indices:index:0*
Tindices0	*
Tparams0*#
_output_shapes
:??????????
8Application_Area_2_indicator/to_sparse_input/dense_shapeShape
features_2*
T0*
_output_shapes
:*
out_type0	?
:Application_Area_2_indicator/None_Lookup/LookupTableFindV2LookupTableFindV2Gapplication_area_2_indicator_none_lookup_lookuptablefindv2_table_handle<Application_Area_2_indicator/to_sparse_input/values:output:0Happlication_area_2_indicator_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:??????????
8Application_Area_2_indicator/SparseToDense/default_valueConst*
_output_shapes
: *
dtype0	*
valueB	 R
??????????
*Application_Area_2_indicator/SparseToDenseSparseToDense<Application_Area_2_indicator/to_sparse_input/indices:index:0AApplication_Area_2_indicator/to_sparse_input/dense_shape:output:0CApplication_Area_2_indicator/None_Lookup/LookupTableFindV2:values:0AApplication_Area_2_indicator/SparseToDense/default_value:output:0*
T0	*
Tindices0	*'
_output_shapes
:?????????o
*Application_Area_2_indicator/one_hot/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??q
,Application_Area_2_indicator/one_hot/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    l
*Application_Area_2_indicator/one_hot/depthConst*
_output_shapes
: *
dtype0*
value	B :"?
$Application_Area_2_indicator/one_hotOneHot2Application_Area_2_indicator/SparseToDense:dense:03Application_Area_2_indicator/one_hot/depth:output:03Application_Area_2_indicator/one_hot/Const:output:05Application_Area_2_indicator/one_hot/Const_1:output:0*
T0*+
_output_shapes
:?????????"?
2Application_Area_2_indicator/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
??????????
 Application_Area_2_indicator/SumSum-Application_Area_2_indicator/one_hot:output:0;Application_Area_2_indicator/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:?????????"{
"Application_Area_2_indicator/ShapeShape)Application_Area_2_indicator/Sum:output:0*
T0*
_output_shapes
:z
0Application_Area_2_indicator/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: |
2Application_Area_2_indicator/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:|
2Application_Area_2_indicator/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
*Application_Area_2_indicator/strided_sliceStridedSlice+Application_Area_2_indicator/Shape:output:09Application_Area_2_indicator/strided_slice/stack:output:0;Application_Area_2_indicator/strided_slice/stack_1:output:0;Application_Area_2_indicator/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskn
,Application_Area_2_indicator/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :"?
*Application_Area_2_indicator/Reshape/shapePack3Application_Area_2_indicator/strided_slice:output:05Application_Area_2_indicator/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
$Application_Area_2_indicator/ReshapeReshape)Application_Area_2_indicator/Sum:output:03Application_Area_2_indicator/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????"w
6Cowork_Abroad_indicator/to_sparse_input/ignore_value/xConst*
_output_shapes
: *
dtype0*
valueB B ?
0Cowork_Abroad_indicator/to_sparse_input/NotEqualNotEqual
features_4?Cowork_Abroad_indicator/to_sparse_input/ignore_value/x:output:0*
T0*'
_output_shapes
:??????????
/Cowork_Abroad_indicator/to_sparse_input/indicesWhere4Cowork_Abroad_indicator/to_sparse_input/NotEqual:z:0*'
_output_shapes
:??????????
.Cowork_Abroad_indicator/to_sparse_input/valuesGatherNd
features_47Cowork_Abroad_indicator/to_sparse_input/indices:index:0*
Tindices0	*
Tparams0*#
_output_shapes
:?????????}
3Cowork_Abroad_indicator/to_sparse_input/dense_shapeShape
features_4*
T0*
_output_shapes
:*
out_type0	?
5Cowork_Abroad_indicator/None_Lookup/LookupTableFindV2LookupTableFindV2Bcowork_abroad_indicator_none_lookup_lookuptablefindv2_table_handle7Cowork_Abroad_indicator/to_sparse_input/values:output:0Ccowork_abroad_indicator_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:?????????~
3Cowork_Abroad_indicator/SparseToDense/default_valueConst*
_output_shapes
: *
dtype0	*
valueB	 R
??????????
%Cowork_Abroad_indicator/SparseToDenseSparseToDense7Cowork_Abroad_indicator/to_sparse_input/indices:index:0<Cowork_Abroad_indicator/to_sparse_input/dense_shape:output:0>Cowork_Abroad_indicator/None_Lookup/LookupTableFindV2:values:0<Cowork_Abroad_indicator/SparseToDense/default_value:output:0*
T0	*
Tindices0	*'
_output_shapes
:?????????j
%Cowork_Abroad_indicator/one_hot/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??l
'Cowork_Abroad_indicator/one_hot/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    g
%Cowork_Abroad_indicator/one_hot/depthConst*
_output_shapes
: *
dtype0*
value	B :?
Cowork_Abroad_indicator/one_hotOneHot-Cowork_Abroad_indicator/SparseToDense:dense:0.Cowork_Abroad_indicator/one_hot/depth:output:0.Cowork_Abroad_indicator/one_hot/Const:output:00Cowork_Abroad_indicator/one_hot/Const_1:output:0*
T0*+
_output_shapes
:??????????
-Cowork_Abroad_indicator/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
??????????
Cowork_Abroad_indicator/SumSum(Cowork_Abroad_indicator/one_hot:output:06Cowork_Abroad_indicator/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:?????????q
Cowork_Abroad_indicator/ShapeShape$Cowork_Abroad_indicator/Sum:output:0*
T0*
_output_shapes
:u
+Cowork_Abroad_indicator/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: w
-Cowork_Abroad_indicator/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:w
-Cowork_Abroad_indicator/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
%Cowork_Abroad_indicator/strided_sliceStridedSlice&Cowork_Abroad_indicator/Shape:output:04Cowork_Abroad_indicator/strided_slice/stack:output:06Cowork_Abroad_indicator/strided_slice/stack_1:output:06Cowork_Abroad_indicator/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maski
'Cowork_Abroad_indicator/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
%Cowork_Abroad_indicator/Reshape/shapePack.Cowork_Abroad_indicator/strided_slice:output:00Cowork_Abroad_indicator/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
Cowork_Abroad_indicator/ReshapeReshape$Cowork_Abroad_indicator/Sum:output:0.Cowork_Abroad_indicator/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????t
3Cowork_Cor_indicator/to_sparse_input/ignore_value/xConst*
_output_shapes
: *
dtype0*
valueB B ?
-Cowork_Cor_indicator/to_sparse_input/NotEqualNotEqual
features_5<Cowork_Cor_indicator/to_sparse_input/ignore_value/x:output:0*
T0*'
_output_shapes
:??????????
,Cowork_Cor_indicator/to_sparse_input/indicesWhere1Cowork_Cor_indicator/to_sparse_input/NotEqual:z:0*'
_output_shapes
:??????????
+Cowork_Cor_indicator/to_sparse_input/valuesGatherNd
features_54Cowork_Cor_indicator/to_sparse_input/indices:index:0*
Tindices0	*
Tparams0*#
_output_shapes
:?????????z
0Cowork_Cor_indicator/to_sparse_input/dense_shapeShape
features_5*
T0*
_output_shapes
:*
out_type0	?
2Cowork_Cor_indicator/None_Lookup/LookupTableFindV2LookupTableFindV2?cowork_cor_indicator_none_lookup_lookuptablefindv2_table_handle4Cowork_Cor_indicator/to_sparse_input/values:output:0@cowork_cor_indicator_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:?????????{
0Cowork_Cor_indicator/SparseToDense/default_valueConst*
_output_shapes
: *
dtype0	*
valueB	 R
??????????
"Cowork_Cor_indicator/SparseToDenseSparseToDense4Cowork_Cor_indicator/to_sparse_input/indices:index:09Cowork_Cor_indicator/to_sparse_input/dense_shape:output:0;Cowork_Cor_indicator/None_Lookup/LookupTableFindV2:values:09Cowork_Cor_indicator/SparseToDense/default_value:output:0*
T0	*
Tindices0	*'
_output_shapes
:?????????g
"Cowork_Cor_indicator/one_hot/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??i
$Cowork_Cor_indicator/one_hot/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    d
"Cowork_Cor_indicator/one_hot/depthConst*
_output_shapes
: *
dtype0*
value	B :?
Cowork_Cor_indicator/one_hotOneHot*Cowork_Cor_indicator/SparseToDense:dense:0+Cowork_Cor_indicator/one_hot/depth:output:0+Cowork_Cor_indicator/one_hot/Const:output:0-Cowork_Cor_indicator/one_hot/Const_1:output:0*
T0*+
_output_shapes
:?????????}
*Cowork_Cor_indicator/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
??????????
Cowork_Cor_indicator/SumSum%Cowork_Cor_indicator/one_hot:output:03Cowork_Cor_indicator/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:?????????k
Cowork_Cor_indicator/ShapeShape!Cowork_Cor_indicator/Sum:output:0*
T0*
_output_shapes
:r
(Cowork_Cor_indicator/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: t
*Cowork_Cor_indicator/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:t
*Cowork_Cor_indicator/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
"Cowork_Cor_indicator/strided_sliceStridedSlice#Cowork_Cor_indicator/Shape:output:01Cowork_Cor_indicator/strided_slice/stack:output:03Cowork_Cor_indicator/strided_slice/stack_1:output:03Cowork_Cor_indicator/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
$Cowork_Cor_indicator/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
"Cowork_Cor_indicator/Reshape/shapePack+Cowork_Cor_indicator/strided_slice:output:0-Cowork_Cor_indicator/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
Cowork_Cor_indicator/ReshapeReshape!Cowork_Cor_indicator/Sum:output:0+Cowork_Cor_indicator/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????u
4Cowork_Inst_indicator/to_sparse_input/ignore_value/xConst*
_output_shapes
: *
dtype0*
valueB B ?
.Cowork_Inst_indicator/to_sparse_input/NotEqualNotEqual
features_6=Cowork_Inst_indicator/to_sparse_input/ignore_value/x:output:0*
T0*'
_output_shapes
:??????????
-Cowork_Inst_indicator/to_sparse_input/indicesWhere2Cowork_Inst_indicator/to_sparse_input/NotEqual:z:0*'
_output_shapes
:??????????
,Cowork_Inst_indicator/to_sparse_input/valuesGatherNd
features_65Cowork_Inst_indicator/to_sparse_input/indices:index:0*
Tindices0	*
Tparams0*#
_output_shapes
:?????????{
1Cowork_Inst_indicator/to_sparse_input/dense_shapeShape
features_6*
T0*
_output_shapes
:*
out_type0	?
3Cowork_Inst_indicator/None_Lookup/LookupTableFindV2LookupTableFindV2@cowork_inst_indicator_none_lookup_lookuptablefindv2_table_handle5Cowork_Inst_indicator/to_sparse_input/values:output:0Acowork_inst_indicator_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:?????????|
1Cowork_Inst_indicator/SparseToDense/default_valueConst*
_output_shapes
: *
dtype0	*
valueB	 R
??????????
#Cowork_Inst_indicator/SparseToDenseSparseToDense5Cowork_Inst_indicator/to_sparse_input/indices:index:0:Cowork_Inst_indicator/to_sparse_input/dense_shape:output:0<Cowork_Inst_indicator/None_Lookup/LookupTableFindV2:values:0:Cowork_Inst_indicator/SparseToDense/default_value:output:0*
T0	*
Tindices0	*'
_output_shapes
:?????????h
#Cowork_Inst_indicator/one_hot/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??j
%Cowork_Inst_indicator/one_hot/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    e
#Cowork_Inst_indicator/one_hot/depthConst*
_output_shapes
: *
dtype0*
value	B :?
Cowork_Inst_indicator/one_hotOneHot+Cowork_Inst_indicator/SparseToDense:dense:0,Cowork_Inst_indicator/one_hot/depth:output:0,Cowork_Inst_indicator/one_hot/Const:output:0.Cowork_Inst_indicator/one_hot/Const_1:output:0*
T0*+
_output_shapes
:?????????~
+Cowork_Inst_indicator/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
??????????
Cowork_Inst_indicator/SumSum&Cowork_Inst_indicator/one_hot:output:04Cowork_Inst_indicator/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:?????????m
Cowork_Inst_indicator/ShapeShape"Cowork_Inst_indicator/Sum:output:0*
T0*
_output_shapes
:s
)Cowork_Inst_indicator/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: u
+Cowork_Inst_indicator/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+Cowork_Inst_indicator/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
#Cowork_Inst_indicator/strided_sliceStridedSlice$Cowork_Inst_indicator/Shape:output:02Cowork_Inst_indicator/strided_slice/stack:output:04Cowork_Inst_indicator/strided_slice/stack_1:output:04Cowork_Inst_indicator/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskg
%Cowork_Inst_indicator/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
#Cowork_Inst_indicator/Reshape/shapePack,Cowork_Inst_indicator/strided_slice:output:0.Cowork_Inst_indicator/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
Cowork_Inst_indicator/ReshapeReshape"Cowork_Inst_indicator/Sum:output:0,Cowork_Inst_indicator/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????t
3Cowork_Uni_indicator/to_sparse_input/ignore_value/xConst*
_output_shapes
: *
dtype0*
valueB B ?
-Cowork_Uni_indicator/to_sparse_input/NotEqualNotEqual
features_7<Cowork_Uni_indicator/to_sparse_input/ignore_value/x:output:0*
T0*'
_output_shapes
:??????????
,Cowork_Uni_indicator/to_sparse_input/indicesWhere1Cowork_Uni_indicator/to_sparse_input/NotEqual:z:0*'
_output_shapes
:??????????
+Cowork_Uni_indicator/to_sparse_input/valuesGatherNd
features_74Cowork_Uni_indicator/to_sparse_input/indices:index:0*
Tindices0	*
Tparams0*#
_output_shapes
:?????????z
0Cowork_Uni_indicator/to_sparse_input/dense_shapeShape
features_7*
T0*
_output_shapes
:*
out_type0	?
2Cowork_Uni_indicator/None_Lookup/LookupTableFindV2LookupTableFindV2?cowork_uni_indicator_none_lookup_lookuptablefindv2_table_handle4Cowork_Uni_indicator/to_sparse_input/values:output:0@cowork_uni_indicator_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:?????????{
0Cowork_Uni_indicator/SparseToDense/default_valueConst*
_output_shapes
: *
dtype0	*
valueB	 R
??????????
"Cowork_Uni_indicator/SparseToDenseSparseToDense4Cowork_Uni_indicator/to_sparse_input/indices:index:09Cowork_Uni_indicator/to_sparse_input/dense_shape:output:0;Cowork_Uni_indicator/None_Lookup/LookupTableFindV2:values:09Cowork_Uni_indicator/SparseToDense/default_value:output:0*
T0	*
Tindices0	*'
_output_shapes
:?????????g
"Cowork_Uni_indicator/one_hot/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??i
$Cowork_Uni_indicator/one_hot/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    d
"Cowork_Uni_indicator/one_hot/depthConst*
_output_shapes
: *
dtype0*
value	B :?
Cowork_Uni_indicator/one_hotOneHot*Cowork_Uni_indicator/SparseToDense:dense:0+Cowork_Uni_indicator/one_hot/depth:output:0+Cowork_Uni_indicator/one_hot/Const:output:0-Cowork_Uni_indicator/one_hot/Const_1:output:0*
T0*+
_output_shapes
:?????????}
*Cowork_Uni_indicator/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
??????????
Cowork_Uni_indicator/SumSum%Cowork_Uni_indicator/one_hot:output:03Cowork_Uni_indicator/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:?????????k
Cowork_Uni_indicator/ShapeShape!Cowork_Uni_indicator/Sum:output:0*
T0*
_output_shapes
:r
(Cowork_Uni_indicator/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: t
*Cowork_Uni_indicator/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:t
*Cowork_Uni_indicator/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
"Cowork_Uni_indicator/strided_sliceStridedSlice#Cowork_Uni_indicator/Shape:output:01Cowork_Uni_indicator/strided_slice/stack:output:03Cowork_Uni_indicator/strided_slice/stack_1:output:03Cowork_Uni_indicator/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
$Cowork_Uni_indicator/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
"Cowork_Uni_indicator/Reshape/shapePack+Cowork_Uni_indicator/strided_slice:output:0-Cowork_Uni_indicator/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
Cowork_Uni_indicator/ReshapeReshape!Cowork_Uni_indicator/Sum:output:0+Cowork_Uni_indicator/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????t
3Cowork_etc_indicator/to_sparse_input/ignore_value/xConst*
_output_shapes
: *
dtype0*
valueB B ?
-Cowork_etc_indicator/to_sparse_input/NotEqualNotEqual
features_8<Cowork_etc_indicator/to_sparse_input/ignore_value/x:output:0*
T0*'
_output_shapes
:??????????
,Cowork_etc_indicator/to_sparse_input/indicesWhere1Cowork_etc_indicator/to_sparse_input/NotEqual:z:0*'
_output_shapes
:??????????
+Cowork_etc_indicator/to_sparse_input/valuesGatherNd
features_84Cowork_etc_indicator/to_sparse_input/indices:index:0*
Tindices0	*
Tparams0*#
_output_shapes
:?????????z
0Cowork_etc_indicator/to_sparse_input/dense_shapeShape
features_8*
T0*
_output_shapes
:*
out_type0	?
2Cowork_etc_indicator/None_Lookup/LookupTableFindV2LookupTableFindV2?cowork_etc_indicator_none_lookup_lookuptablefindv2_table_handle4Cowork_etc_indicator/to_sparse_input/values:output:0@cowork_etc_indicator_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:?????????{
0Cowork_etc_indicator/SparseToDense/default_valueConst*
_output_shapes
: *
dtype0	*
valueB	 R
??????????
"Cowork_etc_indicator/SparseToDenseSparseToDense4Cowork_etc_indicator/to_sparse_input/indices:index:09Cowork_etc_indicator/to_sparse_input/dense_shape:output:0;Cowork_etc_indicator/None_Lookup/LookupTableFindV2:values:09Cowork_etc_indicator/SparseToDense/default_value:output:0*
T0	*
Tindices0	*'
_output_shapes
:?????????g
"Cowork_etc_indicator/one_hot/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??i
$Cowork_etc_indicator/one_hot/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    d
"Cowork_etc_indicator/one_hot/depthConst*
_output_shapes
: *
dtype0*
value	B :?
Cowork_etc_indicator/one_hotOneHot*Cowork_etc_indicator/SparseToDense:dense:0+Cowork_etc_indicator/one_hot/depth:output:0+Cowork_etc_indicator/one_hot/Const:output:0-Cowork_etc_indicator/one_hot/Const_1:output:0*
T0*+
_output_shapes
:?????????}
*Cowork_etc_indicator/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
??????????
Cowork_etc_indicator/SumSum%Cowork_etc_indicator/one_hot:output:03Cowork_etc_indicator/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:?????????k
Cowork_etc_indicator/ShapeShape!Cowork_etc_indicator/Sum:output:0*
T0*
_output_shapes
:r
(Cowork_etc_indicator/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: t
*Cowork_etc_indicator/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:t
*Cowork_etc_indicator/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
"Cowork_etc_indicator/strided_sliceStridedSlice#Cowork_etc_indicator/Shape:output:01Cowork_etc_indicator/strided_slice/stack:output:03Cowork_etc_indicator/strided_slice/stack_1:output:03Cowork_etc_indicator/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
$Cowork_etc_indicator/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
"Cowork_etc_indicator/Reshape/shapePack+Cowork_etc_indicator/strided_slice:output:0-Cowork_etc_indicator/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
Cowork_etc_indicator/ReshapeReshape!Cowork_etc_indicator/Sum:output:0+Cowork_etc_indicator/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????
4Econ_Social_indicator/to_sparse_input/ignore_value/xConst*
_output_shapes
: *
dtype0*
valueB :
??????????
2Econ_Social_indicator/to_sparse_input/ignore_valueCast=Econ_Social_indicator/to_sparse_input/ignore_value/x:output:0*

DstT0	*

SrcT0*
_output_shapes
: ?
.Econ_Social_indicator/to_sparse_input/NotEqualNotEqual
features_96Econ_Social_indicator/to_sparse_input/ignore_value:y:0*
T0	*'
_output_shapes
:??????????
-Econ_Social_indicator/to_sparse_input/indicesWhere2Econ_Social_indicator/to_sparse_input/NotEqual:z:0*'
_output_shapes
:??????????
,Econ_Social_indicator/to_sparse_input/valuesGatherNd
features_95Econ_Social_indicator/to_sparse_input/indices:index:0*
Tindices0	*
Tparams0	*#
_output_shapes
:?????????{
1Econ_Social_indicator/to_sparse_input/dense_shapeShape
features_9*
T0	*
_output_shapes
:*
out_type0	?
3Econ_Social_indicator/None_Lookup/LookupTableFindV2LookupTableFindV2@econ_social_indicator_none_lookup_lookuptablefindv2_table_handle5Econ_Social_indicator/to_sparse_input/values:output:0Aecon_social_indicator_none_lookup_lookuptablefindv2_default_value*	
Tin0	*

Tout0	*#
_output_shapes
:?????????|
1Econ_Social_indicator/SparseToDense/default_valueConst*
_output_shapes
: *
dtype0	*
valueB	 R
??????????
#Econ_Social_indicator/SparseToDenseSparseToDense5Econ_Social_indicator/to_sparse_input/indices:index:0:Econ_Social_indicator/to_sparse_input/dense_shape:output:0<Econ_Social_indicator/None_Lookup/LookupTableFindV2:values:0:Econ_Social_indicator/SparseToDense/default_value:output:0*
T0	*
Tindices0	*'
_output_shapes
:?????????h
#Econ_Social_indicator/one_hot/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??j
%Econ_Social_indicator/one_hot/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    e
#Econ_Social_indicator/one_hot/depthConst*
_output_shapes
: *
dtype0*
value	B :?
Econ_Social_indicator/one_hotOneHot+Econ_Social_indicator/SparseToDense:dense:0,Econ_Social_indicator/one_hot/depth:output:0,Econ_Social_indicator/one_hot/Const:output:0.Econ_Social_indicator/one_hot/Const_1:output:0*
T0*+
_output_shapes
:?????????~
+Econ_Social_indicator/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
??????????
Econ_Social_indicator/SumSum&Econ_Social_indicator/one_hot:output:04Econ_Social_indicator/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:?????????m
Econ_Social_indicator/ShapeShape"Econ_Social_indicator/Sum:output:0*
T0*
_output_shapes
:s
)Econ_Social_indicator/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: u
+Econ_Social_indicator/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+Econ_Social_indicator/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
#Econ_Social_indicator/strided_sliceStridedSlice$Econ_Social_indicator/Shape:output:02Econ_Social_indicator/strided_slice/stack:output:04Econ_Social_indicator/strided_slice/stack_1:output:04Econ_Social_indicator/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskg
%Econ_Social_indicator/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
#Econ_Social_indicator/Reshape/shapePack,Econ_Social_indicator/strided_slice:output:0.Econ_Social_indicator/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
Econ_Social_indicator/ReshapeReshape"Econ_Social_indicator/Sum:output:0,Econ_Social_indicator/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????~
3Green_Tech_indicator/to_sparse_input/ignore_value/xConst*
_output_shapes
: *
dtype0*
valueB :
??????????
1Green_Tech_indicator/to_sparse_input/ignore_valueCast<Green_Tech_indicator/to_sparse_input/ignore_value/x:output:0*

DstT0	*

SrcT0*
_output_shapes
: ?
-Green_Tech_indicator/to_sparse_input/NotEqualNotEqualfeatures_105Green_Tech_indicator/to_sparse_input/ignore_value:y:0*
T0	*'
_output_shapes
:??????????
,Green_Tech_indicator/to_sparse_input/indicesWhere1Green_Tech_indicator/to_sparse_input/NotEqual:z:0*'
_output_shapes
:??????????
+Green_Tech_indicator/to_sparse_input/valuesGatherNdfeatures_104Green_Tech_indicator/to_sparse_input/indices:index:0*
Tindices0	*
Tparams0	*#
_output_shapes
:?????????{
0Green_Tech_indicator/to_sparse_input/dense_shapeShapefeatures_10*
T0	*
_output_shapes
:*
out_type0	?
2Green_Tech_indicator/None_Lookup/LookupTableFindV2LookupTableFindV2?green_tech_indicator_none_lookup_lookuptablefindv2_table_handle4Green_Tech_indicator/to_sparse_input/values:output:0@green_tech_indicator_none_lookup_lookuptablefindv2_default_value*	
Tin0	*

Tout0	*#
_output_shapes
:?????????{
0Green_Tech_indicator/SparseToDense/default_valueConst*
_output_shapes
: *
dtype0	*
valueB	 R
??????????
"Green_Tech_indicator/SparseToDenseSparseToDense4Green_Tech_indicator/to_sparse_input/indices:index:09Green_Tech_indicator/to_sparse_input/dense_shape:output:0;Green_Tech_indicator/None_Lookup/LookupTableFindV2:values:09Green_Tech_indicator/SparseToDense/default_value:output:0*
T0	*
Tindices0	*'
_output_shapes
:?????????g
"Green_Tech_indicator/one_hot/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??i
$Green_Tech_indicator/one_hot/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    d
"Green_Tech_indicator/one_hot/depthConst*
_output_shapes
: *
dtype0*
value	B :<?
Green_Tech_indicator/one_hotOneHot*Green_Tech_indicator/SparseToDense:dense:0+Green_Tech_indicator/one_hot/depth:output:0+Green_Tech_indicator/one_hot/Const:output:0-Green_Tech_indicator/one_hot/Const_1:output:0*
T0*+
_output_shapes
:?????????<}
*Green_Tech_indicator/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
??????????
Green_Tech_indicator/SumSum%Green_Tech_indicator/one_hot:output:03Green_Tech_indicator/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:?????????<k
Green_Tech_indicator/ShapeShape!Green_Tech_indicator/Sum:output:0*
T0*
_output_shapes
:r
(Green_Tech_indicator/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: t
*Green_Tech_indicator/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:t
*Green_Tech_indicator/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
"Green_Tech_indicator/strided_sliceStridedSlice#Green_Tech_indicator/Shape:output:01Green_Tech_indicator/strided_slice/stack:output:03Green_Tech_indicator/strided_slice/stack_1:output:03Green_Tech_indicator/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
$Green_Tech_indicator/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :<?
"Green_Tech_indicator/Reshape/shapePack+Green_Tech_indicator/strided_slice:output:0-Green_Tech_indicator/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
Green_Tech_indicator/ReshapeReshape!Green_Tech_indicator/Sum:output:0+Green_Tech_indicator/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????<M
Log_Duration/ShapeShapefeatures_11*
T0*
_output_shapes
:j
 Log_Duration/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: l
"Log_Duration/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:l
"Log_Duration/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
Log_Duration/strided_sliceStridedSliceLog_Duration/Shape:output:0)Log_Duration/strided_slice/stack:output:0+Log_Duration/strided_slice/stack_1:output:0+Log_Duration/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask^
Log_Duration/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
Log_Duration/Reshape/shapePack#Log_Duration/strided_slice:output:0%Log_Duration/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
Log_Duration/ReshapeReshapefeatures_11#Log_Duration/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????M
Log_RnD_Fund/ShapeShapefeatures_12*
T0*
_output_shapes
:j
 Log_RnD_Fund/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: l
"Log_RnD_Fund/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:l
"Log_RnD_Fund/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
Log_RnD_Fund/strided_sliceStridedSliceLog_RnD_Fund/Shape:output:0)Log_RnD_Fund/strided_slice/stack:output:0+Log_RnD_Fund/strided_slice/stack_1:output:0+Log_RnD_Fund/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask^
Log_RnD_Fund/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
Log_RnD_Fund/Reshape/shapePack#Log_RnD_Fund/strided_slice:output:0%Log_RnD_Fund/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
Log_RnD_Fund/ReshapeReshapefeatures_12#Log_RnD_Fund/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????~
3Multi_Year_indicator/to_sparse_input/ignore_value/xConst*
_output_shapes
: *
dtype0*
valueB :
??????????
1Multi_Year_indicator/to_sparse_input/ignore_valueCast<Multi_Year_indicator/to_sparse_input/ignore_value/x:output:0*

DstT0	*

SrcT0*
_output_shapes
: ?
-Multi_Year_indicator/to_sparse_input/NotEqualNotEqualfeatures_135Multi_Year_indicator/to_sparse_input/ignore_value:y:0*
T0	*'
_output_shapes
:??????????
,Multi_Year_indicator/to_sparse_input/indicesWhere1Multi_Year_indicator/to_sparse_input/NotEqual:z:0*'
_output_shapes
:??????????
+Multi_Year_indicator/to_sparse_input/valuesGatherNdfeatures_134Multi_Year_indicator/to_sparse_input/indices:index:0*
Tindices0	*
Tparams0	*#
_output_shapes
:?????????{
0Multi_Year_indicator/to_sparse_input/dense_shapeShapefeatures_13*
T0	*
_output_shapes
:*
out_type0	?
2Multi_Year_indicator/None_Lookup/LookupTableFindV2LookupTableFindV2?multi_year_indicator_none_lookup_lookuptablefindv2_table_handle4Multi_Year_indicator/to_sparse_input/values:output:0@multi_year_indicator_none_lookup_lookuptablefindv2_default_value*	
Tin0	*

Tout0	*#
_output_shapes
:?????????{
0Multi_Year_indicator/SparseToDense/default_valueConst*
_output_shapes
: *
dtype0	*
valueB	 R
??????????
"Multi_Year_indicator/SparseToDenseSparseToDense4Multi_Year_indicator/to_sparse_input/indices:index:09Multi_Year_indicator/to_sparse_input/dense_shape:output:0;Multi_Year_indicator/None_Lookup/LookupTableFindV2:values:09Multi_Year_indicator/SparseToDense/default_value:output:0*
T0	*
Tindices0	*'
_output_shapes
:?????????g
"Multi_Year_indicator/one_hot/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??i
$Multi_Year_indicator/one_hot/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    d
"Multi_Year_indicator/one_hot/depthConst*
_output_shapes
: *
dtype0*
value	B :?
Multi_Year_indicator/one_hotOneHot*Multi_Year_indicator/SparseToDense:dense:0+Multi_Year_indicator/one_hot/depth:output:0+Multi_Year_indicator/one_hot/Const:output:0-Multi_Year_indicator/one_hot/Const_1:output:0*
T0*+
_output_shapes
:?????????}
*Multi_Year_indicator/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
??????????
Multi_Year_indicator/SumSum%Multi_Year_indicator/one_hot:output:03Multi_Year_indicator/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:?????????k
Multi_Year_indicator/ShapeShape!Multi_Year_indicator/Sum:output:0*
T0*
_output_shapes
:r
(Multi_Year_indicator/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: t
*Multi_Year_indicator/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:t
*Multi_Year_indicator/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
"Multi_Year_indicator/strided_sliceStridedSlice#Multi_Year_indicator/Shape:output:01Multi_Year_indicator/strided_slice/stack:output:03Multi_Year_indicator/strided_slice/stack_1:output:03Multi_Year_indicator/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
$Multi_Year_indicator/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
"Multi_Year_indicator/Reshape/shapePack+Multi_Year_indicator/strided_slice:output:0-Multi_Year_indicator/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
Multi_Year_indicator/ReshapeReshape!Multi_Year_indicator/Sum:output:0+Multi_Year_indicator/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????M
N_Patent_App/ShapeShapefeatures_14*
T0*
_output_shapes
:j
 N_Patent_App/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: l
"N_Patent_App/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:l
"N_Patent_App/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
N_Patent_App/strided_sliceStridedSliceN_Patent_App/Shape:output:0)N_Patent_App/strided_slice/stack:output:0+N_Patent_App/strided_slice/stack_1:output:0+N_Patent_App/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask^
N_Patent_App/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
N_Patent_App/Reshape/shapePack#N_Patent_App/strided_slice:output:0%N_Patent_App/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
N_Patent_App/ReshapeReshapefeatures_14#N_Patent_App/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????M
N_Patent_Reg/ShapeShapefeatures_15*
T0*
_output_shapes
:j
 N_Patent_Reg/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: l
"N_Patent_Reg/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:l
"N_Patent_Reg/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
N_Patent_Reg/strided_sliceStridedSliceN_Patent_Reg/Shape:output:0)N_Patent_Reg/strided_slice/stack:output:0+N_Patent_Reg/strided_slice/stack_1:output:0+N_Patent_Reg/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask^
N_Patent_Reg/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
N_Patent_Reg/Reshape/shapePack#N_Patent_Reg/strided_slice:output:0%N_Patent_Reg/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
N_Patent_Reg/ReshapeReshapefeatures_15#N_Patent_Reg/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????S
N_of_Korean_Patent/ShapeShapefeatures_16*
T0*
_output_shapes
:p
&N_of_Korean_Patent/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: r
(N_of_Korean_Patent/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(N_of_Korean_Patent/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
 N_of_Korean_Patent/strided_sliceStridedSlice!N_of_Korean_Patent/Shape:output:0/N_of_Korean_Patent/strided_slice/stack:output:01N_of_Korean_Patent/strided_slice/stack_1:output:01N_of_Korean_Patent/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskd
"N_of_Korean_Patent/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
 N_of_Korean_Patent/Reshape/shapePack)N_of_Korean_Patent/strided_slice:output:0+N_of_Korean_Patent/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
N_of_Korean_Patent/ReshapeReshapefeatures_16)N_of_Korean_Patent/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????K
N_of_Paper/ShapeShapefeatures_17*
T0*
_output_shapes
:h
N_of_Paper/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: j
 N_of_Paper/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:j
 N_of_Paper/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
N_of_Paper/strided_sliceStridedSliceN_of_Paper/Shape:output:0'N_of_Paper/strided_slice/stack:output:0)N_of_Paper/strided_slice/stack_1:output:0)N_of_Paper/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask\
N_of_Paper/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
N_of_Paper/Reshape/shapePack!N_of_Paper/strided_slice:output:0#N_of_Paper/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:
N_of_Paper/ReshapeReshapefeatures_17!N_of_Paper/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????I
N_of_SCI/ShapeShapefeatures_18*
T0*
_output_shapes
:f
N_of_SCI/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: h
N_of_SCI/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:h
N_of_SCI/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
N_of_SCI/strided_sliceStridedSliceN_of_SCI/Shape:output:0%N_of_SCI/strided_slice/stack:output:0'N_of_SCI/strided_slice/stack_1:output:0'N_of_SCI/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskZ
N_of_SCI/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
N_of_SCI/Reshape/shapePackN_of_SCI/strided_slice:output:0!N_of_SCI/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:{
N_of_SCI/ReshapeReshapefeatures_18N_of_SCI/Reshape/shape:output:0*
T0*'
_output_shapes
:??????????
<National_Strategy_2_indicator/to_sparse_input/ignore_value/xConst*
_output_shapes
: *
dtype0*
valueB :
??????????
:National_Strategy_2_indicator/to_sparse_input/ignore_valueCastENational_Strategy_2_indicator/to_sparse_input/ignore_value/x:output:0*

DstT0	*

SrcT0*
_output_shapes
: ?
6National_Strategy_2_indicator/to_sparse_input/NotEqualNotEqualfeatures_19>National_Strategy_2_indicator/to_sparse_input/ignore_value:y:0*
T0	*'
_output_shapes
:??????????
5National_Strategy_2_indicator/to_sparse_input/indicesWhere:National_Strategy_2_indicator/to_sparse_input/NotEqual:z:0*'
_output_shapes
:??????????
4National_Strategy_2_indicator/to_sparse_input/valuesGatherNdfeatures_19=National_Strategy_2_indicator/to_sparse_input/indices:index:0*
Tindices0	*
Tparams0	*#
_output_shapes
:??????????
9National_Strategy_2_indicator/to_sparse_input/dense_shapeShapefeatures_19*
T0	*
_output_shapes
:*
out_type0	?
;National_Strategy_2_indicator/None_Lookup/LookupTableFindV2LookupTableFindV2Hnational_strategy_2_indicator_none_lookup_lookuptablefindv2_table_handle=National_Strategy_2_indicator/to_sparse_input/values:output:0Inational_strategy_2_indicator_none_lookup_lookuptablefindv2_default_value*	
Tin0	*

Tout0	*#
_output_shapes
:??????????
9National_Strategy_2_indicator/SparseToDense/default_valueConst*
_output_shapes
: *
dtype0	*
valueB	 R
??????????
+National_Strategy_2_indicator/SparseToDenseSparseToDense=National_Strategy_2_indicator/to_sparse_input/indices:index:0BNational_Strategy_2_indicator/to_sparse_input/dense_shape:output:0DNational_Strategy_2_indicator/None_Lookup/LookupTableFindV2:values:0BNational_Strategy_2_indicator/SparseToDense/default_value:output:0*
T0	*
Tindices0	*'
_output_shapes
:?????????p
+National_Strategy_2_indicator/one_hot/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??r
-National_Strategy_2_indicator/one_hot/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    m
+National_Strategy_2_indicator/one_hot/depthConst*
_output_shapes
: *
dtype0*
value	B :?
%National_Strategy_2_indicator/one_hotOneHot3National_Strategy_2_indicator/SparseToDense:dense:04National_Strategy_2_indicator/one_hot/depth:output:04National_Strategy_2_indicator/one_hot/Const:output:06National_Strategy_2_indicator/one_hot/Const_1:output:0*
T0*+
_output_shapes
:??????????
3National_Strategy_2_indicator/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
??????????
!National_Strategy_2_indicator/SumSum.National_Strategy_2_indicator/one_hot:output:0<National_Strategy_2_indicator/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:?????????}
#National_Strategy_2_indicator/ShapeShape*National_Strategy_2_indicator/Sum:output:0*
T0*
_output_shapes
:{
1National_Strategy_2_indicator/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: }
3National_Strategy_2_indicator/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:}
3National_Strategy_2_indicator/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
+National_Strategy_2_indicator/strided_sliceStridedSlice,National_Strategy_2_indicator/Shape:output:0:National_Strategy_2_indicator/strided_slice/stack:output:0<National_Strategy_2_indicator/strided_slice/stack_1:output:0<National_Strategy_2_indicator/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_masko
-National_Strategy_2_indicator/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
+National_Strategy_2_indicator/Reshape/shapePack4National_Strategy_2_indicator/strided_slice:output:06National_Strategy_2_indicator/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
%National_Strategy_2_indicator/ReshapeReshape*National_Strategy_2_indicator/Sum:output:04National_Strategy_2_indicator/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????{
0RnD_Org_indicator/to_sparse_input/ignore_value/xConst*
_output_shapes
: *
dtype0*
valueB :
??????????
.RnD_Org_indicator/to_sparse_input/ignore_valueCast9RnD_Org_indicator/to_sparse_input/ignore_value/x:output:0*

DstT0	*

SrcT0*
_output_shapes
: ?
*RnD_Org_indicator/to_sparse_input/NotEqualNotEqualfeatures_202RnD_Org_indicator/to_sparse_input/ignore_value:y:0*
T0	*'
_output_shapes
:??????????
)RnD_Org_indicator/to_sparse_input/indicesWhere.RnD_Org_indicator/to_sparse_input/NotEqual:z:0*'
_output_shapes
:??????????
(RnD_Org_indicator/to_sparse_input/valuesGatherNdfeatures_201RnD_Org_indicator/to_sparse_input/indices:index:0*
Tindices0	*
Tparams0	*#
_output_shapes
:?????????x
-RnD_Org_indicator/to_sparse_input/dense_shapeShapefeatures_20*
T0	*
_output_shapes
:*
out_type0	?
/RnD_Org_indicator/None_Lookup/LookupTableFindV2LookupTableFindV2<rnd_org_indicator_none_lookup_lookuptablefindv2_table_handle1RnD_Org_indicator/to_sparse_input/values:output:0=rnd_org_indicator_none_lookup_lookuptablefindv2_default_value*	
Tin0	*

Tout0	*#
_output_shapes
:?????????x
-RnD_Org_indicator/SparseToDense/default_valueConst*
_output_shapes
: *
dtype0	*
valueB	 R
??????????
RnD_Org_indicator/SparseToDenseSparseToDense1RnD_Org_indicator/to_sparse_input/indices:index:06RnD_Org_indicator/to_sparse_input/dense_shape:output:08RnD_Org_indicator/None_Lookup/LookupTableFindV2:values:06RnD_Org_indicator/SparseToDense/default_value:output:0*
T0	*
Tindices0	*'
_output_shapes
:?????????d
RnD_Org_indicator/one_hot/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??f
!RnD_Org_indicator/one_hot/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    a
RnD_Org_indicator/one_hot/depthConst*
_output_shapes
: *
dtype0*
value	B :?
RnD_Org_indicator/one_hotOneHot'RnD_Org_indicator/SparseToDense:dense:0(RnD_Org_indicator/one_hot/depth:output:0(RnD_Org_indicator/one_hot/Const:output:0*RnD_Org_indicator/one_hot/Const_1:output:0*
T0*+
_output_shapes
:?????????z
'RnD_Org_indicator/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
??????????
RnD_Org_indicator/SumSum"RnD_Org_indicator/one_hot:output:00RnD_Org_indicator/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:?????????e
RnD_Org_indicator/ShapeShapeRnD_Org_indicator/Sum:output:0*
T0*
_output_shapes
:o
%RnD_Org_indicator/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: q
'RnD_Org_indicator/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:q
'RnD_Org_indicator/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
RnD_Org_indicator/strided_sliceStridedSlice RnD_Org_indicator/Shape:output:0.RnD_Org_indicator/strided_slice/stack:output:00RnD_Org_indicator/strided_slice/stack_1:output:00RnD_Org_indicator/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskc
!RnD_Org_indicator/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
RnD_Org_indicator/Reshape/shapePack(RnD_Org_indicator/strided_slice:output:0*RnD_Org_indicator/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
RnD_Org_indicator/ReshapeReshapeRnD_Org_indicator/Sum:output:0(RnD_Org_indicator/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????}
2RnD_Stage_indicator/to_sparse_input/ignore_value/xConst*
_output_shapes
: *
dtype0*
valueB :
??????????
0RnD_Stage_indicator/to_sparse_input/ignore_valueCast;RnD_Stage_indicator/to_sparse_input/ignore_value/x:output:0*

DstT0	*

SrcT0*
_output_shapes
: ?
,RnD_Stage_indicator/to_sparse_input/NotEqualNotEqualfeatures_214RnD_Stage_indicator/to_sparse_input/ignore_value:y:0*
T0	*'
_output_shapes
:??????????
+RnD_Stage_indicator/to_sparse_input/indicesWhere0RnD_Stage_indicator/to_sparse_input/NotEqual:z:0*'
_output_shapes
:??????????
*RnD_Stage_indicator/to_sparse_input/valuesGatherNdfeatures_213RnD_Stage_indicator/to_sparse_input/indices:index:0*
Tindices0	*
Tparams0	*#
_output_shapes
:?????????z
/RnD_Stage_indicator/to_sparse_input/dense_shapeShapefeatures_21*
T0	*
_output_shapes
:*
out_type0	?
1RnD_Stage_indicator/None_Lookup/LookupTableFindV2LookupTableFindV2>rnd_stage_indicator_none_lookup_lookuptablefindv2_table_handle3RnD_Stage_indicator/to_sparse_input/values:output:0?rnd_stage_indicator_none_lookup_lookuptablefindv2_default_value*	
Tin0	*

Tout0	*#
_output_shapes
:?????????z
/RnD_Stage_indicator/SparseToDense/default_valueConst*
_output_shapes
: *
dtype0	*
valueB	 R
??????????
!RnD_Stage_indicator/SparseToDenseSparseToDense3RnD_Stage_indicator/to_sparse_input/indices:index:08RnD_Stage_indicator/to_sparse_input/dense_shape:output:0:RnD_Stage_indicator/None_Lookup/LookupTableFindV2:values:08RnD_Stage_indicator/SparseToDense/default_value:output:0*
T0	*
Tindices0	*'
_output_shapes
:?????????f
!RnD_Stage_indicator/one_hot/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??h
#RnD_Stage_indicator/one_hot/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    c
!RnD_Stage_indicator/one_hot/depthConst*
_output_shapes
: *
dtype0*
value	B :?
RnD_Stage_indicator/one_hotOneHot)RnD_Stage_indicator/SparseToDense:dense:0*RnD_Stage_indicator/one_hot/depth:output:0*RnD_Stage_indicator/one_hot/Const:output:0,RnD_Stage_indicator/one_hot/Const_1:output:0*
T0*+
_output_shapes
:?????????|
)RnD_Stage_indicator/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
??????????
RnD_Stage_indicator/SumSum$RnD_Stage_indicator/one_hot:output:02RnD_Stage_indicator/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:?????????i
RnD_Stage_indicator/ShapeShape RnD_Stage_indicator/Sum:output:0*
T0*
_output_shapes
:q
'RnD_Stage_indicator/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: s
)RnD_Stage_indicator/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:s
)RnD_Stage_indicator/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
!RnD_Stage_indicator/strided_sliceStridedSlice"RnD_Stage_indicator/Shape:output:00RnD_Stage_indicator/strided_slice/stack:output:02RnD_Stage_indicator/strided_slice/stack_1:output:02RnD_Stage_indicator/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maske
#RnD_Stage_indicator/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
!RnD_Stage_indicator/Reshape/shapePack*RnD_Stage_indicator/strided_slice:output:0,RnD_Stage_indicator/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
RnD_Stage_indicator/ReshapeReshape RnD_Stage_indicator/Sum:output:0*RnD_Stage_indicator/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????u
4STP_Code_11_indicator/to_sparse_input/ignore_value/xConst*
_output_shapes
: *
dtype0*
valueB B ?
.STP_Code_11_indicator/to_sparse_input/NotEqualNotEqualfeatures_22=STP_Code_11_indicator/to_sparse_input/ignore_value/x:output:0*
T0*'
_output_shapes
:??????????
-STP_Code_11_indicator/to_sparse_input/indicesWhere2STP_Code_11_indicator/to_sparse_input/NotEqual:z:0*'
_output_shapes
:??????????
,STP_Code_11_indicator/to_sparse_input/valuesGatherNdfeatures_225STP_Code_11_indicator/to_sparse_input/indices:index:0*
Tindices0	*
Tparams0*#
_output_shapes
:?????????|
1STP_Code_11_indicator/to_sparse_input/dense_shapeShapefeatures_22*
T0*
_output_shapes
:*
out_type0	?
3STP_Code_11_indicator/None_Lookup/LookupTableFindV2LookupTableFindV2@stp_code_11_indicator_none_lookup_lookuptablefindv2_table_handle5STP_Code_11_indicator/to_sparse_input/values:output:0Astp_code_11_indicator_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:?????????|
1STP_Code_11_indicator/SparseToDense/default_valueConst*
_output_shapes
: *
dtype0	*
valueB	 R
??????????
#STP_Code_11_indicator/SparseToDenseSparseToDense5STP_Code_11_indicator/to_sparse_input/indices:index:0:STP_Code_11_indicator/to_sparse_input/dense_shape:output:0<STP_Code_11_indicator/None_Lookup/LookupTableFindV2:values:0:STP_Code_11_indicator/SparseToDense/default_value:output:0*
T0	*
Tindices0	*'
_output_shapes
:?????????h
#STP_Code_11_indicator/one_hot/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??j
%STP_Code_11_indicator/one_hot/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    f
#STP_Code_11_indicator/one_hot/depthConst*
_output_shapes
: *
dtype0*
value
B :??
STP_Code_11_indicator/one_hotOneHot+STP_Code_11_indicator/SparseToDense:dense:0,STP_Code_11_indicator/one_hot/depth:output:0,STP_Code_11_indicator/one_hot/Const:output:0.STP_Code_11_indicator/one_hot/Const_1:output:0*
T0*,
_output_shapes
:??????????~
+STP_Code_11_indicator/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
??????????
STP_Code_11_indicator/SumSum&STP_Code_11_indicator/one_hot:output:04STP_Code_11_indicator/Sum/reduction_indices:output:0*
T0*(
_output_shapes
:??????????m
STP_Code_11_indicator/ShapeShape"STP_Code_11_indicator/Sum:output:0*
T0*
_output_shapes
:s
)STP_Code_11_indicator/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: u
+STP_Code_11_indicator/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+STP_Code_11_indicator/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
#STP_Code_11_indicator/strided_sliceStridedSlice$STP_Code_11_indicator/Shape:output:02STP_Code_11_indicator/strided_slice/stack:output:04STP_Code_11_indicator/strided_slice/stack_1:output:04STP_Code_11_indicator/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskh
%STP_Code_11_indicator/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value
B :??
#STP_Code_11_indicator/Reshape/shapePack,STP_Code_11_indicator/strided_slice:output:0.STP_Code_11_indicator/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
STP_Code_11_indicator/ReshapeReshape"STP_Code_11_indicator/Sum:output:0,STP_Code_11_indicator/Reshape/shape:output:0*
T0*(
_output_shapes
:??????????R
STP_Code_1_Weight/ShapeShapefeatures_23*
T0*
_output_shapes
:o
%STP_Code_1_Weight/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: q
'STP_Code_1_Weight/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:q
'STP_Code_1_Weight/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
STP_Code_1_Weight/strided_sliceStridedSlice STP_Code_1_Weight/Shape:output:0.STP_Code_1_Weight/strided_slice/stack:output:00STP_Code_1_Weight/strided_slice/stack_1:output:00STP_Code_1_Weight/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskc
!STP_Code_1_Weight/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
STP_Code_1_Weight/Reshape/shapePack(STP_Code_1_Weight/strided_slice:output:0*STP_Code_1_Weight/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
STP_Code_1_Weight/ReshapeReshapefeatures_23(STP_Code_1_Weight/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????u
4STP_Code_21_indicator/to_sparse_input/ignore_value/xConst*
_output_shapes
: *
dtype0*
valueB B ?
.STP_Code_21_indicator/to_sparse_input/NotEqualNotEqualfeatures_24=STP_Code_21_indicator/to_sparse_input/ignore_value/x:output:0*
T0*'
_output_shapes
:??????????
-STP_Code_21_indicator/to_sparse_input/indicesWhere2STP_Code_21_indicator/to_sparse_input/NotEqual:z:0*'
_output_shapes
:??????????
,STP_Code_21_indicator/to_sparse_input/valuesGatherNdfeatures_245STP_Code_21_indicator/to_sparse_input/indices:index:0*
Tindices0	*
Tparams0*#
_output_shapes
:?????????|
1STP_Code_21_indicator/to_sparse_input/dense_shapeShapefeatures_24*
T0*
_output_shapes
:*
out_type0	?
3STP_Code_21_indicator/None_Lookup/LookupTableFindV2LookupTableFindV2@stp_code_21_indicator_none_lookup_lookuptablefindv2_table_handle5STP_Code_21_indicator/to_sparse_input/values:output:0Astp_code_21_indicator_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:?????????|
1STP_Code_21_indicator/SparseToDense/default_valueConst*
_output_shapes
: *
dtype0	*
valueB	 R
??????????
#STP_Code_21_indicator/SparseToDenseSparseToDense5STP_Code_21_indicator/to_sparse_input/indices:index:0:STP_Code_21_indicator/to_sparse_input/dense_shape:output:0<STP_Code_21_indicator/None_Lookup/LookupTableFindV2:values:0:STP_Code_21_indicator/SparseToDense/default_value:output:0*
T0	*
Tindices0	*'
_output_shapes
:?????????h
#STP_Code_21_indicator/one_hot/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??j
%STP_Code_21_indicator/one_hot/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    f
#STP_Code_21_indicator/one_hot/depthConst*
_output_shapes
: *
dtype0*
value
B :??
STP_Code_21_indicator/one_hotOneHot+STP_Code_21_indicator/SparseToDense:dense:0,STP_Code_21_indicator/one_hot/depth:output:0,STP_Code_21_indicator/one_hot/Const:output:0.STP_Code_21_indicator/one_hot/Const_1:output:0*
T0*,
_output_shapes
:??????????~
+STP_Code_21_indicator/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
??????????
STP_Code_21_indicator/SumSum&STP_Code_21_indicator/one_hot:output:04STP_Code_21_indicator/Sum/reduction_indices:output:0*
T0*(
_output_shapes
:??????????m
STP_Code_21_indicator/ShapeShape"STP_Code_21_indicator/Sum:output:0*
T0*
_output_shapes
:s
)STP_Code_21_indicator/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: u
+STP_Code_21_indicator/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+STP_Code_21_indicator/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
#STP_Code_21_indicator/strided_sliceStridedSlice$STP_Code_21_indicator/Shape:output:02STP_Code_21_indicator/strided_slice/stack:output:04STP_Code_21_indicator/strided_slice/stack_1:output:04STP_Code_21_indicator/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskh
%STP_Code_21_indicator/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value
B :??
#STP_Code_21_indicator/Reshape/shapePack,STP_Code_21_indicator/strided_slice:output:0.STP_Code_21_indicator/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
STP_Code_21_indicator/ReshapeReshape"STP_Code_21_indicator/Sum:output:0,STP_Code_21_indicator/Reshape/shape:output:0*
T0*(
_output_shapes
:??????????R
STP_Code_2_Weight/ShapeShapefeatures_25*
T0*
_output_shapes
:o
%STP_Code_2_Weight/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: q
'STP_Code_2_Weight/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:q
'STP_Code_2_Weight/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
STP_Code_2_Weight/strided_sliceStridedSlice STP_Code_2_Weight/Shape:output:0.STP_Code_2_Weight/strided_slice/stack:output:00STP_Code_2_Weight/strided_slice/stack_1:output:00STP_Code_2_Weight/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskc
!STP_Code_2_Weight/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
STP_Code_2_Weight/Reshape/shapePack(STP_Code_2_Weight/strided_slice:output:0*STP_Code_2_Weight/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
STP_Code_2_Weight/ReshapeReshapefeatures_25(STP_Code_2_Weight/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????z
/SixT_2_indicator/to_sparse_input/ignore_value/xConst*
_output_shapes
: *
dtype0*
valueB :
??????????
-SixT_2_indicator/to_sparse_input/ignore_valueCast8SixT_2_indicator/to_sparse_input/ignore_value/x:output:0*

DstT0	*

SrcT0*
_output_shapes
: ?
)SixT_2_indicator/to_sparse_input/NotEqualNotEqualfeatures_261SixT_2_indicator/to_sparse_input/ignore_value:y:0*
T0	*'
_output_shapes
:??????????
(SixT_2_indicator/to_sparse_input/indicesWhere-SixT_2_indicator/to_sparse_input/NotEqual:z:0*'
_output_shapes
:??????????
'SixT_2_indicator/to_sparse_input/valuesGatherNdfeatures_260SixT_2_indicator/to_sparse_input/indices:index:0*
Tindices0	*
Tparams0	*#
_output_shapes
:?????????w
,SixT_2_indicator/to_sparse_input/dense_shapeShapefeatures_26*
T0	*
_output_shapes
:*
out_type0	?
.SixT_2_indicator/None_Lookup/LookupTableFindV2LookupTableFindV2;sixt_2_indicator_none_lookup_lookuptablefindv2_table_handle0SixT_2_indicator/to_sparse_input/values:output:0<sixt_2_indicator_none_lookup_lookuptablefindv2_default_value*	
Tin0	*

Tout0	*#
_output_shapes
:?????????w
,SixT_2_indicator/SparseToDense/default_valueConst*
_output_shapes
: *
dtype0	*
valueB	 R
??????????
SixT_2_indicator/SparseToDenseSparseToDense0SixT_2_indicator/to_sparse_input/indices:index:05SixT_2_indicator/to_sparse_input/dense_shape:output:07SixT_2_indicator/None_Lookup/LookupTableFindV2:values:05SixT_2_indicator/SparseToDense/default_value:output:0*
T0	*
Tindices0	*'
_output_shapes
:?????????c
SixT_2_indicator/one_hot/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??e
 SixT_2_indicator/one_hot/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    `
SixT_2_indicator/one_hot/depthConst*
_output_shapes
: *
dtype0*
value	B :?
SixT_2_indicator/one_hotOneHot&SixT_2_indicator/SparseToDense:dense:0'SixT_2_indicator/one_hot/depth:output:0'SixT_2_indicator/one_hot/Const:output:0)SixT_2_indicator/one_hot/Const_1:output:0*
T0*+
_output_shapes
:?????????y
&SixT_2_indicator/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
??????????
SixT_2_indicator/SumSum!SixT_2_indicator/one_hot:output:0/SixT_2_indicator/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:?????????c
SixT_2_indicator/ShapeShapeSixT_2_indicator/Sum:output:0*
T0*
_output_shapes
:n
$SixT_2_indicator/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: p
&SixT_2_indicator/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:p
&SixT_2_indicator/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
SixT_2_indicator/strided_sliceStridedSliceSixT_2_indicator/Shape:output:0-SixT_2_indicator/strided_slice/stack:output:0/SixT_2_indicator/strided_slice/stack_1:output:0/SixT_2_indicator/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskb
 SixT_2_indicator/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
SixT_2_indicator/Reshape/shapePack'SixT_2_indicator/strided_slice:output:0)SixT_2_indicator/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
SixT_2_indicator/ReshapeReshapeSixT_2_indicator/Sum:output:0'SixT_2_indicator/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????E

Year/ShapeShapefeatures_27*
T0*
_output_shapes
:b
Year/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: d
Year/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:d
Year/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
Year/strided_sliceStridedSliceYear/Shape:output:0!Year/strided_slice/stack:output:0#Year/strided_slice/stack_1:output:0#Year/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskV
Year/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
Year/Reshape/shapePackYear/strided_slice:output:0Year/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:s
Year/ReshapeReshapefeatures_27Year/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????V
concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
??????????
concatConcatV2*Application_Area_1_Weight/Reshape:output:0-Application_Area_1_indicator/Reshape:output:0*Application_Area_2_Weight/Reshape:output:0-Application_Area_2_indicator/Reshape:output:0(Cowork_Abroad_indicator/Reshape:output:0%Cowork_Cor_indicator/Reshape:output:0&Cowork_Inst_indicator/Reshape:output:0%Cowork_Uni_indicator/Reshape:output:0%Cowork_etc_indicator/Reshape:output:0&Econ_Social_indicator/Reshape:output:0%Green_Tech_indicator/Reshape:output:0Log_Duration/Reshape:output:0Log_RnD_Fund/Reshape:output:0%Multi_Year_indicator/Reshape:output:0N_Patent_App/Reshape:output:0N_Patent_Reg/Reshape:output:0#N_of_Korean_Patent/Reshape:output:0N_of_Paper/Reshape:output:0N_of_SCI/Reshape:output:0.National_Strategy_2_indicator/Reshape:output:0"RnD_Org_indicator/Reshape:output:0$RnD_Stage_indicator/Reshape:output:0&STP_Code_11_indicator/Reshape:output:0"STP_Code_1_Weight/Reshape:output:0&STP_Code_21_indicator/Reshape:output:0"STP_Code_2_Weight/Reshape:output:0!SixT_2_indicator/Reshape:output:0Year/Reshape:output:0concat/axis:output:0*
N*
T0*(
_output_shapes
:??????????_
IdentityIdentityconcat:output:0^NoOp*
T0*(
_output_shapes
:???????????
NoOpNoOp;^Application_Area_1_indicator/None_Lookup/LookupTableFindV2;^Application_Area_2_indicator/None_Lookup/LookupTableFindV26^Cowork_Abroad_indicator/None_Lookup/LookupTableFindV23^Cowork_Cor_indicator/None_Lookup/LookupTableFindV24^Cowork_Inst_indicator/None_Lookup/LookupTableFindV23^Cowork_Uni_indicator/None_Lookup/LookupTableFindV23^Cowork_etc_indicator/None_Lookup/LookupTableFindV24^Econ_Social_indicator/None_Lookup/LookupTableFindV23^Green_Tech_indicator/None_Lookup/LookupTableFindV23^Multi_Year_indicator/None_Lookup/LookupTableFindV2<^National_Strategy_2_indicator/None_Lookup/LookupTableFindV20^RnD_Org_indicator/None_Lookup/LookupTableFindV22^RnD_Stage_indicator/None_Lookup/LookupTableFindV24^STP_Code_11_indicator/None_Lookup/LookupTableFindV24^STP_Code_21_indicator/None_Lookup/LookupTableFindV2/^SixT_2_indicator/None_Lookup/LookupTableFindV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2x
:Application_Area_1_indicator/None_Lookup/LookupTableFindV2:Application_Area_1_indicator/None_Lookup/LookupTableFindV22x
:Application_Area_2_indicator/None_Lookup/LookupTableFindV2:Application_Area_2_indicator/None_Lookup/LookupTableFindV22n
5Cowork_Abroad_indicator/None_Lookup/LookupTableFindV25Cowork_Abroad_indicator/None_Lookup/LookupTableFindV22h
2Cowork_Cor_indicator/None_Lookup/LookupTableFindV22Cowork_Cor_indicator/None_Lookup/LookupTableFindV22j
3Cowork_Inst_indicator/None_Lookup/LookupTableFindV23Cowork_Inst_indicator/None_Lookup/LookupTableFindV22h
2Cowork_Uni_indicator/None_Lookup/LookupTableFindV22Cowork_Uni_indicator/None_Lookup/LookupTableFindV22h
2Cowork_etc_indicator/None_Lookup/LookupTableFindV22Cowork_etc_indicator/None_Lookup/LookupTableFindV22j
3Econ_Social_indicator/None_Lookup/LookupTableFindV23Econ_Social_indicator/None_Lookup/LookupTableFindV22h
2Green_Tech_indicator/None_Lookup/LookupTableFindV22Green_Tech_indicator/None_Lookup/LookupTableFindV22h
2Multi_Year_indicator/None_Lookup/LookupTableFindV22Multi_Year_indicator/None_Lookup/LookupTableFindV22z
;National_Strategy_2_indicator/None_Lookup/LookupTableFindV2;National_Strategy_2_indicator/None_Lookup/LookupTableFindV22b
/RnD_Org_indicator/None_Lookup/LookupTableFindV2/RnD_Org_indicator/None_Lookup/LookupTableFindV22f
1RnD_Stage_indicator/None_Lookup/LookupTableFindV21RnD_Stage_indicator/None_Lookup/LookupTableFindV22j
3STP_Code_11_indicator/None_Lookup/LookupTableFindV23STP_Code_11_indicator/None_Lookup/LookupTableFindV22j
3STP_Code_21_indicator/None_Lookup/LookupTableFindV23STP_Code_21_indicator/None_Lookup/LookupTableFindV22`
.SixT_2_indicator/None_Lookup/LookupTableFindV2.SixT_2_indicator/None_Lookup/LookupTableFindV2:Q M
'
_output_shapes
:?????????
"
_user_specified_name
features:QM
'
_output_shapes
:?????????
"
_user_specified_name
features:QM
'
_output_shapes
:?????????
"
_user_specified_name
features:QM
'
_output_shapes
:?????????
"
_user_specified_name
features:QM
'
_output_shapes
:?????????
"
_user_specified_name
features:QM
'
_output_shapes
:?????????
"
_user_specified_name
features:QM
'
_output_shapes
:?????????
"
_user_specified_name
features:QM
'
_output_shapes
:?????????
"
_user_specified_name
features:QM
'
_output_shapes
:?????????
"
_user_specified_name
features:Q	M
'
_output_shapes
:?????????
"
_user_specified_name
features:Q
M
'
_output_shapes
:?????????
"
_user_specified_name
features:QM
'
_output_shapes
:?????????
"
_user_specified_name
features:QM
'
_output_shapes
:?????????
"
_user_specified_name
features:QM
'
_output_shapes
:?????????
"
_user_specified_name
features:QM
'
_output_shapes
:?????????
"
_user_specified_name
features:QM
'
_output_shapes
:?????????
"
_user_specified_name
features:QM
'
_output_shapes
:?????????
"
_user_specified_name
features:QM
'
_output_shapes
:?????????
"
_user_specified_name
features:QM
'
_output_shapes
:?????????
"
_user_specified_name
features:QM
'
_output_shapes
:?????????
"
_user_specified_name
features:QM
'
_output_shapes
:?????????
"
_user_specified_name
features:QM
'
_output_shapes
:?????????
"
_user_specified_name
features:QM
'
_output_shapes
:?????????
"
_user_specified_name
features:QM
'
_output_shapes
:?????????
"
_user_specified_name
features:QM
'
_output_shapes
:?????????
"
_user_specified_name
features:QM
'
_output_shapes
:?????????
"
_user_specified_name
features:QM
'
_output_shapes
:?????????
"
_user_specified_name
features:QM
'
_output_shapes
:?????????
"
_user_specified_name
features:

_output_shapes
: :

_output_shapes
: :!

_output_shapes
: :#

_output_shapes
: :%

_output_shapes
: :'

_output_shapes
: :)

_output_shapes
: :+

_output_shapes
: :-

_output_shapes
: :/

_output_shapes
: :1

_output_shapes
: :3

_output_shapes
: :5

_output_shapes
: :7

_output_shapes
: :9

_output_shapes
: :;

_output_shapes
: 
?
;
__inference__creator_133673
identity??
hash_tablek

hash_tableHashTableV2*
_output_shapes
: *
	key_dtype0*
shared_name342*
value_dtype0	W
IdentityIdentityhash_table:table_handle:0^NoOp*
T0*
_output_shapes
: S
NoOpNoOp^hash_table*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 2

hash_table
hash_table
?
;
__inference__creator_133871
identity??
hash_tablek

hash_tableHashTableV2*
_output_shapes
: *
	key_dtype0	*
shared_name784*
value_dtype0	W
IdentityIdentityhash_table:table_handle:0^NoOp*
T0*
_output_shapes
: S
NoOpNoOp^hash_table*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 2

hash_table
hash_table
?
-
__inference__destroyer_133938
identityG
ConstConst*
_output_shapes
: *
dtype0*
value	B :E
IdentityIdentityConst:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 
?2
?

$__inference_signature_wrapper_130910
application_area_1
application_area_1_weight
application_area_2
application_area_2_weight
cowork_abroad

cowork_cor
cowork_inst

cowork_uni

cowork_etc
econ_social	

green_tech	
log_duration
log_rnd_fund

multi_year	
n_patent_app
n_patent_reg
n_of_korean_patent

n_of_paper
n_of_sci
national_strategy_2	
rnd_org	
	rnd_stage	
stp_code_11
stp_code_1_weight
stp_code_21
stp_code_2_weight

sixt_2	
year
unknown
	unknown_0	
	unknown_1
	unknown_2	
	unknown_3
	unknown_4	
	unknown_5
	unknown_6	
	unknown_7
	unknown_8	
	unknown_9

unknown_10	

unknown_11

unknown_12	

unknown_13

unknown_14	

unknown_15

unknown_16	

unknown_17

unknown_18	

unknown_19

unknown_20	

unknown_21

unknown_22	

unknown_23

unknown_24	

unknown_25

unknown_26	

unknown_27

unknown_28	

unknown_29

unknown_30	

unknown_31:
??

unknown_32:	?

unknown_33:
??

unknown_34:	?

unknown_35:
??

unknown_36:	?

unknown_37:	?

unknown_38:
identity??StatefulPartitionedCall?	
StatefulPartitionedCallStatefulPartitionedCallapplication_area_1application_area_1_weightapplication_area_2application_area_2_weightcowork_abroad
cowork_corcowork_inst
cowork_uni
cowork_etcecon_social
green_techlog_durationlog_rnd_fund
multi_yearn_patent_appn_patent_regn_of_korean_patent
n_of_papern_of_scinational_strategy_2rnd_org	rnd_stagestp_code_11stp_code_1_weightstp_code_21stp_code_2_weightsixt_2yearunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38*O
TinH
F2D																							*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????**
_read_only_resource_inputs

<=>?@ABC*0
config_proto 

CPU

GPU2*0J 8? **
f%R#
!__inference__wrapped_model_128455o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:[ W
'
_output_shapes
:?????????
,
_user_specified_nameApplication_Area_1:b^
'
_output_shapes
:?????????
3
_user_specified_nameApplication_Area_1_Weight:[W
'
_output_shapes
:?????????
,
_user_specified_nameApplication_Area_2:b^
'
_output_shapes
:?????????
3
_user_specified_nameApplication_Area_2_Weight:VR
'
_output_shapes
:?????????
'
_user_specified_nameCowork_Abroad:SO
'
_output_shapes
:?????????
$
_user_specified_name
Cowork_Cor:TP
'
_output_shapes
:?????????
%
_user_specified_nameCowork_Inst:SO
'
_output_shapes
:?????????
$
_user_specified_name
Cowork_Uni:SO
'
_output_shapes
:?????????
$
_user_specified_name
Cowork_etc:T	P
'
_output_shapes
:?????????
%
_user_specified_nameEcon_Social:S
O
'
_output_shapes
:?????????
$
_user_specified_name
Green_Tech:UQ
'
_output_shapes
:?????????
&
_user_specified_nameLog_Duration:UQ
'
_output_shapes
:?????????
&
_user_specified_nameLog_RnD_Fund:SO
'
_output_shapes
:?????????
$
_user_specified_name
Multi_Year:UQ
'
_output_shapes
:?????????
&
_user_specified_nameN_Patent_App:UQ
'
_output_shapes
:?????????
&
_user_specified_nameN_Patent_Reg:[W
'
_output_shapes
:?????????
,
_user_specified_nameN_of_Korean_Patent:SO
'
_output_shapes
:?????????
$
_user_specified_name
N_of_Paper:QM
'
_output_shapes
:?????????
"
_user_specified_name
N_of_SCI:\X
'
_output_shapes
:?????????
-
_user_specified_nameNational_Strategy_2:PL
'
_output_shapes
:?????????
!
_user_specified_name	RnD_Org:RN
'
_output_shapes
:?????????
#
_user_specified_name	RnD_Stage:TP
'
_output_shapes
:?????????
%
_user_specified_nameSTP_Code_11:ZV
'
_output_shapes
:?????????
+
_user_specified_nameSTP_Code_1_Weight:TP
'
_output_shapes
:?????????
%
_user_specified_nameSTP_Code_21:ZV
'
_output_shapes
:?????????
+
_user_specified_nameSTP_Code_2_Weight:OK
'
_output_shapes
:?????????
 
_user_specified_nameSixT_2:MI
'
_output_shapes
:?????????

_user_specified_nameYear:

_output_shapes
: :

_output_shapes
: :!

_output_shapes
: :#

_output_shapes
: :%

_output_shapes
: :'

_output_shapes
: :)

_output_shapes
: :+

_output_shapes
: :-

_output_shapes
: :/

_output_shapes
: :1

_output_shapes
: :3

_output_shapes
: :5

_output_shapes
: :7

_output_shapes
: :9

_output_shapes
: :;

_output_shapes
: 
?
?
(__inference_dense_1_layer_call_fn_133545

inputs
unknown:
??
	unknown_0:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_129137p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:??????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
-
__inference__destroyer_133758
identityG
ConstConst*
_output_shapes
: *
dtype0*
value	B :E
IdentityIdentityConst:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 
?M
?
__inference__traced_save_134301
file_prefix6
2savev2_sequential_dense_kernel_read_readvariableop4
0savev2_sequential_dense_bias_read_readvariableop8
4savev2_sequential_dense_1_kernel_read_readvariableop6
2savev2_sequential_dense_1_bias_read_readvariableop8
4savev2_sequential_dense_2_kernel_read_readvariableop6
2savev2_sequential_dense_2_bias_read_readvariableop8
4savev2_sequential_dense_3_kernel_read_readvariableop6
2savev2_sequential_dense_3_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop-
)savev2_true_positives_read_readvariableop.
*savev2_false_positives_read_readvariableop.
*savev2_false_negatives_read_readvariableop3
/savev2_weights_intermediate_read_readvariableop=
9savev2_adam_sequential_dense_kernel_m_read_readvariableop;
7savev2_adam_sequential_dense_bias_m_read_readvariableop?
;savev2_adam_sequential_dense_1_kernel_m_read_readvariableop=
9savev2_adam_sequential_dense_1_bias_m_read_readvariableop?
;savev2_adam_sequential_dense_2_kernel_m_read_readvariableop=
9savev2_adam_sequential_dense_2_bias_m_read_readvariableop?
;savev2_adam_sequential_dense_3_kernel_m_read_readvariableop=
9savev2_adam_sequential_dense_3_bias_m_read_readvariableop=
9savev2_adam_sequential_dense_kernel_v_read_readvariableop;
7savev2_adam_sequential_dense_bias_v_read_readvariableop?
;savev2_adam_sequential_dense_1_kernel_v_read_readvariableop=
9savev2_adam_sequential_dense_1_bias_v_read_readvariableop?
;savev2_adam_sequential_dense_2_kernel_v_read_readvariableop=
9savev2_adam_sequential_dense_2_bias_v_read_readvariableop?
;savev2_adam_sequential_dense_3_kernel_v_read_readvariableop=
9savev2_adam_sequential_dense_3_bias_v_read_readvariableop
savev2_const_48

identity_1??MergeV2Checkpointsw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part?
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : ?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: ?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:$*
dtype0*?
value?B?$B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/1/true_positives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/1/false_positives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/1/false_negatives/.ATTRIBUTES/VARIABLE_VALUEBCkeras_api/metrics/1/weights_intermediate/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:$*
dtype0*[
valueRBP$B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B ?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:02savev2_sequential_dense_kernel_read_readvariableop0savev2_sequential_dense_bias_read_readvariableop4savev2_sequential_dense_1_kernel_read_readvariableop2savev2_sequential_dense_1_bias_read_readvariableop4savev2_sequential_dense_2_kernel_read_readvariableop2savev2_sequential_dense_2_bias_read_readvariableop4savev2_sequential_dense_3_kernel_read_readvariableop2savev2_sequential_dense_3_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop)savev2_true_positives_read_readvariableop*savev2_false_positives_read_readvariableop*savev2_false_negatives_read_readvariableop/savev2_weights_intermediate_read_readvariableop9savev2_adam_sequential_dense_kernel_m_read_readvariableop7savev2_adam_sequential_dense_bias_m_read_readvariableop;savev2_adam_sequential_dense_1_kernel_m_read_readvariableop9savev2_adam_sequential_dense_1_bias_m_read_readvariableop;savev2_adam_sequential_dense_2_kernel_m_read_readvariableop9savev2_adam_sequential_dense_2_bias_m_read_readvariableop;savev2_adam_sequential_dense_3_kernel_m_read_readvariableop9savev2_adam_sequential_dense_3_bias_m_read_readvariableop9savev2_adam_sequential_dense_kernel_v_read_readvariableop7savev2_adam_sequential_dense_bias_v_read_readvariableop;savev2_adam_sequential_dense_1_kernel_v_read_readvariableop9savev2_adam_sequential_dense_1_bias_v_read_readvariableop;savev2_adam_sequential_dense_2_kernel_v_read_readvariableop9savev2_adam_sequential_dense_2_bias_v_read_readvariableop;savev2_adam_sequential_dense_3_kernel_v_read_readvariableop9savev2_adam_sequential_dense_3_bias_v_read_readvariableopsavev2_const_48"/device:CPU:0*
_output_shapes
 *2
dtypes(
&2$	?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*?
_input_shapes?
?: :
??:?:
??:?:
??:?:	?:: : : : : : : : : : : :
??:?:
??:?:
??:?:	?::
??:?:
??:?:
??:?:	?:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:&"
 
_output_shapes
:
??:!

_output_shapes	
:?:&"
 
_output_shapes
:
??:!

_output_shapes	
:?:&"
 
_output_shapes
:
??:!

_output_shapes	
:?:%!

_output_shapes
:	?: 

_output_shapes
::	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :&"
 
_output_shapes
:
??:!

_output_shapes	
:?:&"
 
_output_shapes
:
??:!

_output_shapes	
:?:&"
 
_output_shapes
:
??:!

_output_shapes	
:?:%!

_output_shapes
:	?: 

_output_shapes
::&"
 
_output_shapes
:
??:!

_output_shapes	
:?:&"
 
_output_shapes
:
??:!

_output_shapes	
:?:& "
 
_output_shapes
:
??:!!

_output_shapes	
:?:%"!

_output_shapes
:	?: #

_output_shapes
::$

_output_shapes
: 
?
?
__inference__initializer_1338972
.table_init817_lookuptableimportv2_table_handle*
&table_init817_lookuptableimportv2_keys,
(table_init817_lookuptableimportv2_values	
identity??!table_init817/LookupTableImportV2?
!table_init817/LookupTableImportV2LookupTableImportV2.table_init817_lookuptableimportv2_table_handle&table_init817_lookuptableimportv2_keys(table_init817_lookuptableimportv2_values*	
Tin0*

Tout0	*
_output_shapes
 G
ConstConst*
_output_shapes
: *
dtype0*
value	B :L
IdentityIdentityConst:output:0^NoOp*
T0*
_output_shapes
: j
NoOpNoOp"^table_init817/LookupTableImportV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*#
_input_shapes
: :?:?2F
!table_init817/LookupTableImportV2!table_init817/LookupTableImportV2:!

_output_shapes	
:?:!

_output_shapes	
:?
?
a
C__inference_dropout_layer_call_and_return_conditional_losses_129124

inputs

identity_1O
IdentityIdentityinputs*
T0*(
_output_shapes
:??????????\

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:??????????"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
__inference__initializer_1337172
.table_init409_lookuptableimportv2_table_handle*
&table_init409_lookuptableimportv2_keys,
(table_init409_lookuptableimportv2_values	
identity??!table_init409/LookupTableImportV2?
!table_init409/LookupTableImportV2LookupTableImportV2.table_init409_lookuptableimportv2_table_handle&table_init409_lookuptableimportv2_keys(table_init409_lookuptableimportv2_values*	
Tin0*

Tout0	*
_output_shapes
 G
ConstConst*
_output_shapes
: *
dtype0*
value	B :L
IdentityIdentityConst:output:0^NoOp*
T0*
_output_shapes
: j
NoOpNoOp"^table_init409/LookupTableImportV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*!
_input_shapes
: ::2F
!table_init409/LookupTableImportV2!table_init409/LookupTableImportV2: 

_output_shapes
:: 

_output_shapes
:
?
;
__inference__creator_133817
identity??
hash_tablek

hash_tableHashTableV2*
_output_shapes
: *
	key_dtype0	*
shared_name636*
value_dtype0	W
IdentityIdentityhash_table:table_handle:0^NoOp*
T0*
_output_shapes
: S
NoOpNoOp^hash_table*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 2

hash_table
hash_table
?V
?
F__inference_sequential_layer_call_and_return_conditional_losses_130790
application_area_1
application_area_1_weight
application_area_2
application_area_2_weight
cowork_abroad

cowork_cor
cowork_inst

cowork_uni

cowork_etc
econ_social	

green_tech	
log_duration
log_rnd_fund

multi_year	
n_patent_app
n_patent_reg
n_of_korean_patent

n_of_paper
n_of_sci
national_strategy_2	
rnd_org	
	rnd_stage	
stp_code_11
stp_code_1_weight
stp_code_21
stp_code_2_weight

sixt_2	
year
dense_features_130701
dense_features_130703	
dense_features_130705
dense_features_130707	
dense_features_130709
dense_features_130711	
dense_features_130713
dense_features_130715	
dense_features_130717
dense_features_130719	
dense_features_130721
dense_features_130723	
dense_features_130725
dense_features_130727	
dense_features_130729
dense_features_130731	
dense_features_130733
dense_features_130735	
dense_features_130737
dense_features_130739	
dense_features_130741
dense_features_130743	
dense_features_130745
dense_features_130747	
dense_features_130749
dense_features_130751	
dense_features_130753
dense_features_130755	
dense_features_130757
dense_features_130759	
dense_features_130761
dense_features_130763	 
dense_130766:
??
dense_130768:	?"
dense_1_130772:
??
dense_1_130774:	?"
dense_2_130778:
??
dense_2_130780:	?!
dense_3_130784:	?
dense_3_130786:
identity??dense/StatefulPartitionedCall?dense_1/StatefulPartitionedCall?dense_2/StatefulPartitionedCall?dense_3/StatefulPartitionedCall?&dense_features/StatefulPartitionedCall?dropout/StatefulPartitionedCall?!dropout_1/StatefulPartitionedCall?!dropout_2/StatefulPartitionedCall?
&dense_features/StatefulPartitionedCallStatefulPartitionedCallapplication_area_1application_area_1_weightapplication_area_2application_area_2_weightcowork_abroad
cowork_corcowork_inst
cowork_uni
cowork_etcecon_social
green_techlog_durationlog_rnd_fund
multi_yearn_patent_appn_patent_regn_of_korean_patent
n_of_papern_of_scinational_strategy_2rnd_org	rnd_stagestp_code_11stp_code_1_weightstp_code_21stp_code_2_weightsixt_2yeardense_features_130701dense_features_130703dense_features_130705dense_features_130707dense_features_130709dense_features_130711dense_features_130713dense_features_130715dense_features_130717dense_features_130719dense_features_130721dense_features_130723dense_features_130725dense_features_130727dense_features_130729dense_features_130731dense_features_130733dense_features_130735dense_features_130737dense_features_130739dense_features_130741dense_features_130743dense_features_130745dense_features_130747dense_features_130749dense_features_130751dense_features_130753dense_features_130755dense_features_130757dense_features_130759dense_features_130761dense_features_130763*G
Tin@
>2<																							*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_dense_features_layer_call_and_return_conditional_losses_130030?
dense/StatefulPartitionedCallStatefulPartitionedCall/dense_features/StatefulPartitionedCall:output:0dense_130766dense_130768*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_129113?
dropout/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_dropout_layer_call_and_return_conditional_losses_129371?
dense_1/StatefulPartitionedCallStatefulPartitionedCall(dropout/StatefulPartitionedCall:output:0dense_1_130772dense_1_130774*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_129137?
!dropout_1/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0 ^dropout/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_dropout_1_layer_call_and_return_conditional_losses_129338?
dense_2/StatefulPartitionedCallStatefulPartitionedCall*dropout_1/StatefulPartitionedCall:output:0dense_2_130778dense_2_130780*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_dense_2_layer_call_and_return_conditional_losses_129161?
!dropout_2/StatefulPartitionedCallStatefulPartitionedCall(dense_2/StatefulPartitionedCall:output:0"^dropout_1/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_dropout_2_layer_call_and_return_conditional_losses_129305?
dense_3/StatefulPartitionedCallStatefulPartitionedCall*dropout_2/StatefulPartitionedCall:output:0dense_3_130784dense_3_130786*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_dense_3_layer_call_and_return_conditional_losses_129185w
IdentityIdentity(dense_3/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall'^dense_features/StatefulPartitionedCall ^dropout/StatefulPartitionedCall"^dropout_1/StatefulPartitionedCall"^dropout_2/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall2P
&dense_features/StatefulPartitionedCall&dense_features/StatefulPartitionedCall2B
dropout/StatefulPartitionedCalldropout/StatefulPartitionedCall2F
!dropout_1/StatefulPartitionedCall!dropout_1/StatefulPartitionedCall2F
!dropout_2/StatefulPartitionedCall!dropout_2/StatefulPartitionedCall:[ W
'
_output_shapes
:?????????
,
_user_specified_nameApplication_Area_1:b^
'
_output_shapes
:?????????
3
_user_specified_nameApplication_Area_1_Weight:[W
'
_output_shapes
:?????????
,
_user_specified_nameApplication_Area_2:b^
'
_output_shapes
:?????????
3
_user_specified_nameApplication_Area_2_Weight:VR
'
_output_shapes
:?????????
'
_user_specified_nameCowork_Abroad:SO
'
_output_shapes
:?????????
$
_user_specified_name
Cowork_Cor:TP
'
_output_shapes
:?????????
%
_user_specified_nameCowork_Inst:SO
'
_output_shapes
:?????????
$
_user_specified_name
Cowork_Uni:SO
'
_output_shapes
:?????????
$
_user_specified_name
Cowork_etc:T	P
'
_output_shapes
:?????????
%
_user_specified_nameEcon_Social:S
O
'
_output_shapes
:?????????
$
_user_specified_name
Green_Tech:UQ
'
_output_shapes
:?????????
&
_user_specified_nameLog_Duration:UQ
'
_output_shapes
:?????????
&
_user_specified_nameLog_RnD_Fund:SO
'
_output_shapes
:?????????
$
_user_specified_name
Multi_Year:UQ
'
_output_shapes
:?????????
&
_user_specified_nameN_Patent_App:UQ
'
_output_shapes
:?????????
&
_user_specified_nameN_Patent_Reg:[W
'
_output_shapes
:?????????
,
_user_specified_nameN_of_Korean_Patent:SO
'
_output_shapes
:?????????
$
_user_specified_name
N_of_Paper:QM
'
_output_shapes
:?????????
"
_user_specified_name
N_of_SCI:\X
'
_output_shapes
:?????????
-
_user_specified_nameNational_Strategy_2:PL
'
_output_shapes
:?????????
!
_user_specified_name	RnD_Org:RN
'
_output_shapes
:?????????
#
_user_specified_name	RnD_Stage:TP
'
_output_shapes
:?????????
%
_user_specified_nameSTP_Code_11:ZV
'
_output_shapes
:?????????
+
_user_specified_nameSTP_Code_1_Weight:TP
'
_output_shapes
:?????????
%
_user_specified_nameSTP_Code_21:ZV
'
_output_shapes
:?????????
+
_user_specified_nameSTP_Code_2_Weight:OK
'
_output_shapes
:?????????
 
_user_specified_nameSixT_2:MI
'
_output_shapes
:?????????

_user_specified_nameYear:

_output_shapes
: :

_output_shapes
: :!

_output_shapes
: :#

_output_shapes
: :%

_output_shapes
: :'

_output_shapes
: :)

_output_shapes
: :+

_output_shapes
: :-

_output_shapes
: :/

_output_shapes
: :1

_output_shapes
: :3

_output_shapes
: :5

_output_shapes
: :7

_output_shapes
: :9

_output_shapes
: :;

_output_shapes
: 
?N
?
F__inference_sequential_layer_call_and_return_conditional_losses_129192

inputs
inputs_1
inputs_2
inputs_3
inputs_4
inputs_5
inputs_6
inputs_7
inputs_8
inputs_9	
	inputs_10	
	inputs_11
	inputs_12
	inputs_13	
	inputs_14
	inputs_15
	inputs_16
	inputs_17
	inputs_18
	inputs_19	
	inputs_20	
	inputs_21	
	inputs_22
	inputs_23
	inputs_24
	inputs_25
	inputs_26	
	inputs_27
dense_features_129037
dense_features_129039	
dense_features_129041
dense_features_129043	
dense_features_129045
dense_features_129047	
dense_features_129049
dense_features_129051	
dense_features_129053
dense_features_129055	
dense_features_129057
dense_features_129059	
dense_features_129061
dense_features_129063	
dense_features_129065
dense_features_129067	
dense_features_129069
dense_features_129071	
dense_features_129073
dense_features_129075	
dense_features_129077
dense_features_129079	
dense_features_129081
dense_features_129083	
dense_features_129085
dense_features_129087	
dense_features_129089
dense_features_129091	
dense_features_129093
dense_features_129095	
dense_features_129097
dense_features_129099	 
dense_129114:
??
dense_129116:	?"
dense_1_129138:
??
dense_1_129140:	?"
dense_2_129162:
??
dense_2_129164:	?!
dense_3_129186:	?
dense_3_129188:
identity??dense/StatefulPartitionedCall?dense_1/StatefulPartitionedCall?dense_2/StatefulPartitionedCall?dense_3/StatefulPartitionedCall?&dense_features/StatefulPartitionedCall?
&dense_features/StatefulPartitionedCallStatefulPartitionedCallinputsinputs_1inputs_2inputs_3inputs_4inputs_5inputs_6inputs_7inputs_8inputs_9	inputs_10	inputs_11	inputs_12	inputs_13	inputs_14	inputs_15	inputs_16	inputs_17	inputs_18	inputs_19	inputs_20	inputs_21	inputs_22	inputs_23	inputs_24	inputs_25	inputs_26	inputs_27dense_features_129037dense_features_129039dense_features_129041dense_features_129043dense_features_129045dense_features_129047dense_features_129049dense_features_129051dense_features_129053dense_features_129055dense_features_129057dense_features_129059dense_features_129061dense_features_129063dense_features_129065dense_features_129067dense_features_129069dense_features_129071dense_features_129073dense_features_129075dense_features_129077dense_features_129079dense_features_129081dense_features_129083dense_features_129085dense_features_129087dense_features_129089dense_features_129091dense_features_129093dense_features_129095dense_features_129097dense_features_129099*G
Tin@
>2<																							*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_dense_features_layer_call_and_return_conditional_losses_129036?
dense/StatefulPartitionedCallStatefulPartitionedCall/dense_features/StatefulPartitionedCall:output:0dense_129114dense_129116*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_129113?
dropout/PartitionedCallPartitionedCall&dense/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_dropout_layer_call_and_return_conditional_losses_129124?
dense_1/StatefulPartitionedCallStatefulPartitionedCall dropout/PartitionedCall:output:0dense_1_129138dense_1_129140*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_129137?
dropout_1/PartitionedCallPartitionedCall(dense_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_dropout_1_layer_call_and_return_conditional_losses_129148?
dense_2/StatefulPartitionedCallStatefulPartitionedCall"dropout_1/PartitionedCall:output:0dense_2_129162dense_2_129164*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_dense_2_layer_call_and_return_conditional_losses_129161?
dropout_2/PartitionedCallPartitionedCall(dense_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_dropout_2_layer_call_and_return_conditional_losses_129172?
dense_3/StatefulPartitionedCallStatefulPartitionedCall"dropout_2/PartitionedCall:output:0dense_3_129186dense_3_129188*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_dense_3_layer_call_and_return_conditional_losses_129185w
IdentityIdentity(dense_3/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall'^dense_features/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall2P
&dense_features/StatefulPartitionedCall&dense_features/StatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:O	K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:O
K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :!

_output_shapes
: :#

_output_shapes
: :%

_output_shapes
: :'

_output_shapes
: :)

_output_shapes
: :+

_output_shapes
: :-

_output_shapes
: :/

_output_shapes
: :1

_output_shapes
: :3

_output_shapes
: :5

_output_shapes
: :7

_output_shapes
: :9

_output_shapes
: :;

_output_shapes
: 
?

?
C__inference_dense_1_layer_call_and_return_conditional_losses_129137

inputs2
matmul_readvariableop_resource:
??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????Q
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:??????????b
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:??????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
;
__inference__creator_133781
identity??
hash_tablek

hash_tableHashTableV2*
_output_shapes
: *
	key_dtype0	*
shared_name548*
value_dtype0	W
IdentityIdentityhash_table:table_handle:0^NoOp*
T0*
_output_shapes
: S
NoOpNoOp^hash_table*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 2

hash_table
hash_table
?	
d
E__inference_dropout_1_layer_call_and_return_conditional_losses_133583

inputs
identity?R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??e
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:??????????C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????p
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????j
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:??????????Z
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:??????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?

?
A__inference_dense_layer_call_and_return_conditional_losses_129113

inputs2
matmul_readvariableop_resource:
??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????Q
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:??????????b
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:??????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
??
?
"__inference__traced_restore_134416
file_prefix<
(assignvariableop_sequential_dense_kernel:
??7
(assignvariableop_1_sequential_dense_bias:	?@
,assignvariableop_2_sequential_dense_1_kernel:
??9
*assignvariableop_3_sequential_dense_1_bias:	?@
,assignvariableop_4_sequential_dense_2_kernel:
??9
*assignvariableop_5_sequential_dense_2_bias:	??
,assignvariableop_6_sequential_dense_3_kernel:	?8
*assignvariableop_7_sequential_dense_3_bias:&
assignvariableop_8_adam_iter:	 (
assignvariableop_9_adam_beta_1: )
assignvariableop_10_adam_beta_2: (
assignvariableop_11_adam_decay: 0
&assignvariableop_12_adam_learning_rate: #
assignvariableop_13_total: #
assignvariableop_14_count: ,
"assignvariableop_15_true_positives: -
#assignvariableop_16_false_positives: -
#assignvariableop_17_false_negatives: 2
(assignvariableop_18_weights_intermediate: F
2assignvariableop_19_adam_sequential_dense_kernel_m:
???
0assignvariableop_20_adam_sequential_dense_bias_m:	?H
4assignvariableop_21_adam_sequential_dense_1_kernel_m:
??A
2assignvariableop_22_adam_sequential_dense_1_bias_m:	?H
4assignvariableop_23_adam_sequential_dense_2_kernel_m:
??A
2assignvariableop_24_adam_sequential_dense_2_bias_m:	?G
4assignvariableop_25_adam_sequential_dense_3_kernel_m:	?@
2assignvariableop_26_adam_sequential_dense_3_bias_m:F
2assignvariableop_27_adam_sequential_dense_kernel_v:
???
0assignvariableop_28_adam_sequential_dense_bias_v:	?H
4assignvariableop_29_adam_sequential_dense_1_kernel_v:
??A
2assignvariableop_30_adam_sequential_dense_1_bias_v:	?H
4assignvariableop_31_adam_sequential_dense_2_kernel_v:
??A
2assignvariableop_32_adam_sequential_dense_2_bias_v:	?G
4assignvariableop_33_adam_sequential_dense_3_kernel_v:	?@
2assignvariableop_34_adam_sequential_dense_3_bias_v:
identity_36??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_22?AssignVariableOp_23?AssignVariableOp_24?AssignVariableOp_25?AssignVariableOp_26?AssignVariableOp_27?AssignVariableOp_28?AssignVariableOp_29?AssignVariableOp_3?AssignVariableOp_30?AssignVariableOp_31?AssignVariableOp_32?AssignVariableOp_33?AssignVariableOp_34?AssignVariableOp_4?AssignVariableOp_5?AssignVariableOp_6?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:$*
dtype0*?
value?B?$B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/1/true_positives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/1/false_positives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/1/false_negatives/.ATTRIBUTES/VARIABLE_VALUEBCkeras_api/metrics/1/weights_intermediate/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:$*
dtype0*[
valueRBP$B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B ?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*?
_output_shapes?
?::::::::::::::::::::::::::::::::::::*2
dtypes(
&2$	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOpAssignVariableOp(assignvariableop_sequential_dense_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_1AssignVariableOp(assignvariableop_1_sequential_dense_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_2AssignVariableOp,assignvariableop_2_sequential_dense_1_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_3AssignVariableOp*assignvariableop_3_sequential_dense_1_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_4AssignVariableOp,assignvariableop_4_sequential_dense_2_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_5AssignVariableOp*assignvariableop_5_sequential_dense_2_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_6AssignVariableOp,assignvariableop_6_sequential_dense_3_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_7AssignVariableOp*assignvariableop_7_sequential_dense_3_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0	*
_output_shapes
:?
AssignVariableOp_8AssignVariableOpassignvariableop_8_adam_iterIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_9AssignVariableOpassignvariableop_9_adam_beta_1Identity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_10AssignVariableOpassignvariableop_10_adam_beta_2Identity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_11AssignVariableOpassignvariableop_11_adam_decayIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_12AssignVariableOp&assignvariableop_12_adam_learning_rateIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_13AssignVariableOpassignvariableop_13_totalIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_14AssignVariableOpassignvariableop_14_countIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_15AssignVariableOp"assignvariableop_15_true_positivesIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_16AssignVariableOp#assignvariableop_16_false_positivesIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_17AssignVariableOp#assignvariableop_17_false_negativesIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_18AssignVariableOp(assignvariableop_18_weights_intermediateIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_19AssignVariableOp2assignvariableop_19_adam_sequential_dense_kernel_mIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_20AssignVariableOp0assignvariableop_20_adam_sequential_dense_bias_mIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_21AssignVariableOp4assignvariableop_21_adam_sequential_dense_1_kernel_mIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_22AssignVariableOp2assignvariableop_22_adam_sequential_dense_1_bias_mIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_23AssignVariableOp4assignvariableop_23_adam_sequential_dense_2_kernel_mIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_24AssignVariableOp2assignvariableop_24_adam_sequential_dense_2_bias_mIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_25AssignVariableOp4assignvariableop_25_adam_sequential_dense_3_kernel_mIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_26AssignVariableOp2assignvariableop_26_adam_sequential_dense_3_bias_mIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_27AssignVariableOp2assignvariableop_27_adam_sequential_dense_kernel_vIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_28AssignVariableOp0assignvariableop_28_adam_sequential_dense_bias_vIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_29AssignVariableOp4assignvariableop_29_adam_sequential_dense_1_kernel_vIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_30AssignVariableOp2assignvariableop_30_adam_sequential_dense_1_bias_vIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_31AssignVariableOp4assignvariableop_31_adam_sequential_dense_2_kernel_vIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_32AssignVariableOp2assignvariableop_32_adam_sequential_dense_2_bias_vIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_33AssignVariableOp4assignvariableop_33_adam_sequential_dense_3_kernel_vIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_34AssignVariableOp2assignvariableop_34_adam_sequential_dense_3_bias_vIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 ?
Identity_35Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_36IdentityIdentity_35:output:0^NoOp_1*
T0*
_output_shapes
: ?
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_36Identity_36:output:0*[
_input_shapesJ
H: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
?
?
__inference_<lambda>_1340342
.table_init747_lookuptableimportv2_table_handle*
&table_init747_lookuptableimportv2_keys	,
(table_init747_lookuptableimportv2_values	
identity??!table_init747/LookupTableImportV2?
!table_init747/LookupTableImportV2LookupTableImportV2.table_init747_lookuptableimportv2_table_handle&table_init747_lookuptableimportv2_keys(table_init747_lookuptableimportv2_values*	
Tin0	*

Tout0	*
_output_shapes
 J
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??L
IdentityIdentityConst:output:0^NoOp*
T0*
_output_shapes
: j
NoOpNoOp"^table_init747/LookupTableImportV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*!
_input_shapes
: ::2F
!table_init747/LookupTableImportV2!table_init747/LookupTableImportV2: 

_output_shapes
:: 

_output_shapes
:
?
F
*__inference_dropout_1_layer_call_fn_133561

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_dropout_1_layer_call_and_return_conditional_losses_129148a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:??????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
__inference_<lambda>_1339542
.table_init341_lookuptableimportv2_table_handle*
&table_init341_lookuptableimportv2_keys,
(table_init341_lookuptableimportv2_values	
identity??!table_init341/LookupTableImportV2?
!table_init341/LookupTableImportV2LookupTableImportV2.table_init341_lookuptableimportv2_table_handle&table_init341_lookuptableimportv2_keys(table_init341_lookuptableimportv2_values*	
Tin0*

Tout0	*
_output_shapes
 J
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??L
IdentityIdentityConst:output:0^NoOp*
T0*
_output_shapes
: j
NoOpNoOp"^table_init341/LookupTableImportV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*!
_input_shapes
: :":"2F
!table_init341/LookupTableImportV2!table_init341/LookupTableImportV2: 

_output_shapes
:": 

_output_shapes
:"
?
c
*__inference_dropout_1_layer_call_fn_133566

inputs
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_dropout_1_layer_call_and_return_conditional_losses_129338p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:??????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:??????????22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
;
__inference__creator_133763
identity??
hash_tablek

hash_tableHashTableV2*
_output_shapes
: *
	key_dtype0*
shared_name512*
value_dtype0	W
IdentityIdentityhash_table:table_handle:0^NoOp*
T0*
_output_shapes
: S
NoOpNoOp^hash_table*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 2

hash_table
hash_table
?
;
__inference__creator_133745
identity??
hash_tablek

hash_tableHashTableV2*
_output_shapes
: *
	key_dtype0*
shared_name478*
value_dtype0	W
IdentityIdentityhash_table:table_handle:0^NoOp*
T0*
_output_shapes
: S
NoOpNoOp^hash_table*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 2

hash_table
hash_table
?3
?

+__inference_sequential_layer_call_fn_130552
application_area_1
application_area_1_weight
application_area_2
application_area_2_weight
cowork_abroad

cowork_cor
cowork_inst

cowork_uni

cowork_etc
econ_social	

green_tech	
log_duration
log_rnd_fund

multi_year	
n_patent_app
n_patent_reg
n_of_korean_patent

n_of_paper
n_of_sci
national_strategy_2	
rnd_org	
	rnd_stage	
stp_code_11
stp_code_1_weight
stp_code_21
stp_code_2_weight

sixt_2	
year
unknown
	unknown_0	
	unknown_1
	unknown_2	
	unknown_3
	unknown_4	
	unknown_5
	unknown_6	
	unknown_7
	unknown_8	
	unknown_9

unknown_10	

unknown_11

unknown_12	

unknown_13

unknown_14	

unknown_15

unknown_16	

unknown_17

unknown_18	

unknown_19

unknown_20	

unknown_21

unknown_22	

unknown_23

unknown_24	

unknown_25

unknown_26	

unknown_27

unknown_28	

unknown_29

unknown_30	

unknown_31:
??

unknown_32:	?

unknown_33:
??

unknown_34:	?

unknown_35:
??

unknown_36:	?

unknown_37:	?

unknown_38:
identity??StatefulPartitionedCall?	
StatefulPartitionedCallStatefulPartitionedCallapplication_area_1application_area_1_weightapplication_area_2application_area_2_weightcowork_abroad
cowork_corcowork_inst
cowork_uni
cowork_etcecon_social
green_techlog_durationlog_rnd_fund
multi_yearn_patent_appn_patent_regn_of_korean_patent
n_of_papern_of_scinational_strategy_2rnd_org	rnd_stagestp_code_11stp_code_1_weightstp_code_21stp_code_2_weightsixt_2yearunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38*O
TinH
F2D																							*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????**
_read_only_resource_inputs

<=>?@ABC*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_sequential_layer_call_and_return_conditional_losses_130357o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:[ W
'
_output_shapes
:?????????
,
_user_specified_nameApplication_Area_1:b^
'
_output_shapes
:?????????
3
_user_specified_nameApplication_Area_1_Weight:[W
'
_output_shapes
:?????????
,
_user_specified_nameApplication_Area_2:b^
'
_output_shapes
:?????????
3
_user_specified_nameApplication_Area_2_Weight:VR
'
_output_shapes
:?????????
'
_user_specified_nameCowork_Abroad:SO
'
_output_shapes
:?????????
$
_user_specified_name
Cowork_Cor:TP
'
_output_shapes
:?????????
%
_user_specified_nameCowork_Inst:SO
'
_output_shapes
:?????????
$
_user_specified_name
Cowork_Uni:SO
'
_output_shapes
:?????????
$
_user_specified_name
Cowork_etc:T	P
'
_output_shapes
:?????????
%
_user_specified_nameEcon_Social:S
O
'
_output_shapes
:?????????
$
_user_specified_name
Green_Tech:UQ
'
_output_shapes
:?????????
&
_user_specified_nameLog_Duration:UQ
'
_output_shapes
:?????????
&
_user_specified_nameLog_RnD_Fund:SO
'
_output_shapes
:?????????
$
_user_specified_name
Multi_Year:UQ
'
_output_shapes
:?????????
&
_user_specified_nameN_Patent_App:UQ
'
_output_shapes
:?????????
&
_user_specified_nameN_Patent_Reg:[W
'
_output_shapes
:?????????
,
_user_specified_nameN_of_Korean_Patent:SO
'
_output_shapes
:?????????
$
_user_specified_name
N_of_Paper:QM
'
_output_shapes
:?????????
"
_user_specified_name
N_of_SCI:\X
'
_output_shapes
:?????????
-
_user_specified_nameNational_Strategy_2:PL
'
_output_shapes
:?????????
!
_user_specified_name	RnD_Org:RN
'
_output_shapes
:?????????
#
_user_specified_name	RnD_Stage:TP
'
_output_shapes
:?????????
%
_user_specified_nameSTP_Code_11:ZV
'
_output_shapes
:?????????
+
_user_specified_nameSTP_Code_1_Weight:TP
'
_output_shapes
:?????????
%
_user_specified_nameSTP_Code_21:ZV
'
_output_shapes
:?????????
+
_user_specified_nameSTP_Code_2_Weight:OK
'
_output_shapes
:?????????
 
_user_specified_nameSixT_2:MI
'
_output_shapes
:?????????

_user_specified_nameYear:

_output_shapes
: :

_output_shapes
: :!

_output_shapes
: :#

_output_shapes
: :%

_output_shapes
: :'

_output_shapes
: :)

_output_shapes
: :+

_output_shapes
: :-

_output_shapes
: :/

_output_shapes
: :1

_output_shapes
: :3

_output_shapes
: :5

_output_shapes
: :7

_output_shapes
: :9

_output_shapes
: :;

_output_shapes
: 
?
?
__inference_<lambda>_1340422
.table_init783_lookuptableimportv2_table_handle*
&table_init783_lookuptableimportv2_keys	,
(table_init783_lookuptableimportv2_values	
identity??!table_init783/LookupTableImportV2?
!table_init783/LookupTableImportV2LookupTableImportV2.table_init783_lookuptableimportv2_table_handle&table_init783_lookuptableimportv2_keys(table_init783_lookuptableimportv2_values*	
Tin0	*

Tout0	*
_output_shapes
 J
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??L
IdentityIdentityConst:output:0^NoOp*
T0*
_output_shapes
: j
NoOpNoOp"^table_init783/LookupTableImportV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*!
_input_shapes
: ::2F
!table_init783/LookupTableImportV2!table_init783/LookupTableImportV2: 

_output_shapes
:: 

_output_shapes
:
?
?
__inference__initializer_1338072
.table_init583_lookuptableimportv2_table_handle*
&table_init583_lookuptableimportv2_keys	,
(table_init583_lookuptableimportv2_values	
identity??!table_init583/LookupTableImportV2?
!table_init583/LookupTableImportV2LookupTableImportV2.table_init583_lookuptableimportv2_table_handle&table_init583_lookuptableimportv2_keys(table_init583_lookuptableimportv2_values*	
Tin0	*

Tout0	*
_output_shapes
 G
ConstConst*
_output_shapes
: *
dtype0*
value	B :L
IdentityIdentityConst:output:0^NoOp*
T0*
_output_shapes
: j
NoOpNoOp"^table_init583/LookupTableImportV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*!
_input_shapes
: :<:<2F
!table_init583/LookupTableImportV2!table_init583/LookupTableImportV2: 

_output_shapes
:<: 

_output_shapes
:<
?
;
__inference__creator_133889
identity??
hash_tablek

hash_tableHashTableV2*
_output_shapes
: *
	key_dtype0*
shared_name818*
value_dtype0	W
IdentityIdentityhash_table:table_handle:0^NoOp*
T0*
_output_shapes
: S
NoOpNoOp^hash_table*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 2

hash_table
hash_table
?
?
__inference_<lambda>_1340662
.table_init903_lookuptableimportv2_table_handle*
&table_init903_lookuptableimportv2_keys	,
(table_init903_lookuptableimportv2_values	
identity??!table_init903/LookupTableImportV2?
!table_init903/LookupTableImportV2LookupTableImportV2.table_init903_lookuptableimportv2_table_handle&table_init903_lookuptableimportv2_keys(table_init903_lookuptableimportv2_values*	
Tin0	*

Tout0	*
_output_shapes
 J
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??L
IdentityIdentityConst:output:0^NoOp*
T0*
_output_shapes
: j
NoOpNoOp"^table_init903/LookupTableImportV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*!
_input_shapes
: ::2F
!table_init903/LookupTableImportV2!table_init903/LookupTableImportV2: 

_output_shapes
:: 

_output_shapes
:
?7
?
+__inference_sequential_layer_call_fn_131134
inputs_application_area_1$
 inputs_application_area_1_weight
inputs_application_area_2$
 inputs_application_area_2_weight
inputs_cowork_abroad
inputs_cowork_cor
inputs_cowork_inst
inputs_cowork_uni
inputs_cowork_etc
inputs_econ_social	
inputs_green_tech	
inputs_log_duration
inputs_log_rnd_fund
inputs_multi_year	
inputs_n_patent_app
inputs_n_patent_reg
inputs_n_of_korean_patent
inputs_n_of_paper
inputs_n_of_sci
inputs_national_strategy_2	
inputs_rnd_org	
inputs_rnd_stage	
inputs_stp_code_11
inputs_stp_code_1_weight
inputs_stp_code_21
inputs_stp_code_2_weight
inputs_sixt_2	
inputs_year
unknown
	unknown_0	
	unknown_1
	unknown_2	
	unknown_3
	unknown_4	
	unknown_5
	unknown_6	
	unknown_7
	unknown_8	
	unknown_9

unknown_10	

unknown_11

unknown_12	

unknown_13

unknown_14	

unknown_15

unknown_16	

unknown_17

unknown_18	

unknown_19

unknown_20	

unknown_21

unknown_22	

unknown_23

unknown_24	

unknown_25

unknown_26	

unknown_27

unknown_28	

unknown_29

unknown_30	

unknown_31:
??

unknown_32:	?

unknown_33:
??

unknown_34:	?

unknown_35:
??

unknown_36:	?

unknown_37:	?

unknown_38:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs_application_area_1 inputs_application_area_1_weightinputs_application_area_2 inputs_application_area_2_weightinputs_cowork_abroadinputs_cowork_corinputs_cowork_instinputs_cowork_uniinputs_cowork_etcinputs_econ_socialinputs_green_techinputs_log_durationinputs_log_rnd_fundinputs_multi_yearinputs_n_patent_appinputs_n_patent_reginputs_n_of_korean_patentinputs_n_of_paperinputs_n_of_sciinputs_national_strategy_2inputs_rnd_orginputs_rnd_stageinputs_stp_code_11inputs_stp_code_1_weightinputs_stp_code_21inputs_stp_code_2_weightinputs_sixt_2inputs_yearunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38*O
TinH
F2D																							*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????**
_read_only_resource_inputs

<=>?@ABC*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_sequential_layer_call_and_return_conditional_losses_130357o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:b ^
'
_output_shapes
:?????????
3
_user_specified_nameinputs/Application_Area_1:ie
'
_output_shapes
:?????????
:
_user_specified_name" inputs/Application_Area_1_Weight:b^
'
_output_shapes
:?????????
3
_user_specified_nameinputs/Application_Area_2:ie
'
_output_shapes
:?????????
:
_user_specified_name" inputs/Application_Area_2_Weight:]Y
'
_output_shapes
:?????????
.
_user_specified_nameinputs/Cowork_Abroad:ZV
'
_output_shapes
:?????????
+
_user_specified_nameinputs/Cowork_Cor:[W
'
_output_shapes
:?????????
,
_user_specified_nameinputs/Cowork_Inst:ZV
'
_output_shapes
:?????????
+
_user_specified_nameinputs/Cowork_Uni:ZV
'
_output_shapes
:?????????
+
_user_specified_nameinputs/Cowork_etc:[	W
'
_output_shapes
:?????????
,
_user_specified_nameinputs/Econ_Social:Z
V
'
_output_shapes
:?????????
+
_user_specified_nameinputs/Green_Tech:\X
'
_output_shapes
:?????????
-
_user_specified_nameinputs/Log_Duration:\X
'
_output_shapes
:?????????
-
_user_specified_nameinputs/Log_RnD_Fund:ZV
'
_output_shapes
:?????????
+
_user_specified_nameinputs/Multi_Year:\X
'
_output_shapes
:?????????
-
_user_specified_nameinputs/N_Patent_App:\X
'
_output_shapes
:?????????
-
_user_specified_nameinputs/N_Patent_Reg:b^
'
_output_shapes
:?????????
3
_user_specified_nameinputs/N_of_Korean_Patent:ZV
'
_output_shapes
:?????????
+
_user_specified_nameinputs/N_of_Paper:XT
'
_output_shapes
:?????????
)
_user_specified_nameinputs/N_of_SCI:c_
'
_output_shapes
:?????????
4
_user_specified_nameinputs/National_Strategy_2:WS
'
_output_shapes
:?????????
(
_user_specified_nameinputs/RnD_Org:YU
'
_output_shapes
:?????????
*
_user_specified_nameinputs/RnD_Stage:[W
'
_output_shapes
:?????????
,
_user_specified_nameinputs/STP_Code_11:a]
'
_output_shapes
:?????????
2
_user_specified_nameinputs/STP_Code_1_Weight:[W
'
_output_shapes
:?????????
,
_user_specified_nameinputs/STP_Code_21:a]
'
_output_shapes
:?????????
2
_user_specified_nameinputs/STP_Code_2_Weight:VR
'
_output_shapes
:?????????
'
_user_specified_nameinputs/SixT_2:TP
'
_output_shapes
:?????????
%
_user_specified_nameinputs/Year:

_output_shapes
: :

_output_shapes
: :!

_output_shapes
: :#

_output_shapes
: :%

_output_shapes
: :'

_output_shapes
: :)

_output_shapes
: :+

_output_shapes
: :-

_output_shapes
: :/

_output_shapes
: :1

_output_shapes
: :3

_output_shapes
: :5

_output_shapes
: :7

_output_shapes
: :9

_output_shapes
: :;

_output_shapes
: 
?
D
(__inference_dropout_layer_call_fn_133514

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_dropout_layer_call_and_return_conditional_losses_129124a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:??????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
ڨ
?-
!__inference__wrapped_model_128455
application_area_1
application_area_1_weight
application_area_2
application_area_2_weight
cowork_abroad

cowork_cor
cowork_inst

cowork_uni

cowork_etc
econ_social	

green_tech	
log_duration
log_rnd_fund

multi_year	
n_patent_app
n_patent_reg
n_of_korean_patent

n_of_paper
n_of_sci
national_strategy_2	
rnd_org	
	rnd_stage	
stp_code_11
stp_code_1_weight
stp_code_21
stp_code_2_weight

sixt_2	
yeare
asequential_dense_features_application_area_1_indicator_none_lookup_lookuptablefindv2_table_handlef
bsequential_dense_features_application_area_1_indicator_none_lookup_lookuptablefindv2_default_value	e
asequential_dense_features_application_area_2_indicator_none_lookup_lookuptablefindv2_table_handlef
bsequential_dense_features_application_area_2_indicator_none_lookup_lookuptablefindv2_default_value	`
\sequential_dense_features_cowork_abroad_indicator_none_lookup_lookuptablefindv2_table_handlea
]sequential_dense_features_cowork_abroad_indicator_none_lookup_lookuptablefindv2_default_value	]
Ysequential_dense_features_cowork_cor_indicator_none_lookup_lookuptablefindv2_table_handle^
Zsequential_dense_features_cowork_cor_indicator_none_lookup_lookuptablefindv2_default_value	^
Zsequential_dense_features_cowork_inst_indicator_none_lookup_lookuptablefindv2_table_handle_
[sequential_dense_features_cowork_inst_indicator_none_lookup_lookuptablefindv2_default_value	]
Ysequential_dense_features_cowork_uni_indicator_none_lookup_lookuptablefindv2_table_handle^
Zsequential_dense_features_cowork_uni_indicator_none_lookup_lookuptablefindv2_default_value	]
Ysequential_dense_features_cowork_etc_indicator_none_lookup_lookuptablefindv2_table_handle^
Zsequential_dense_features_cowork_etc_indicator_none_lookup_lookuptablefindv2_default_value	^
Zsequential_dense_features_econ_social_indicator_none_lookup_lookuptablefindv2_table_handle_
[sequential_dense_features_econ_social_indicator_none_lookup_lookuptablefindv2_default_value	]
Ysequential_dense_features_green_tech_indicator_none_lookup_lookuptablefindv2_table_handle^
Zsequential_dense_features_green_tech_indicator_none_lookup_lookuptablefindv2_default_value	]
Ysequential_dense_features_multi_year_indicator_none_lookup_lookuptablefindv2_table_handle^
Zsequential_dense_features_multi_year_indicator_none_lookup_lookuptablefindv2_default_value	f
bsequential_dense_features_national_strategy_2_indicator_none_lookup_lookuptablefindv2_table_handleg
csequential_dense_features_national_strategy_2_indicator_none_lookup_lookuptablefindv2_default_value	Z
Vsequential_dense_features_rnd_org_indicator_none_lookup_lookuptablefindv2_table_handle[
Wsequential_dense_features_rnd_org_indicator_none_lookup_lookuptablefindv2_default_value	\
Xsequential_dense_features_rnd_stage_indicator_none_lookup_lookuptablefindv2_table_handle]
Ysequential_dense_features_rnd_stage_indicator_none_lookup_lookuptablefindv2_default_value	^
Zsequential_dense_features_stp_code_11_indicator_none_lookup_lookuptablefindv2_table_handle_
[sequential_dense_features_stp_code_11_indicator_none_lookup_lookuptablefindv2_default_value	^
Zsequential_dense_features_stp_code_21_indicator_none_lookup_lookuptablefindv2_table_handle_
[sequential_dense_features_stp_code_21_indicator_none_lookup_lookuptablefindv2_default_value	Y
Usequential_dense_features_sixt_2_indicator_none_lookup_lookuptablefindv2_table_handleZ
Vsequential_dense_features_sixt_2_indicator_none_lookup_lookuptablefindv2_default_value	C
/sequential_dense_matmul_readvariableop_resource:
???
0sequential_dense_biasadd_readvariableop_resource:	?E
1sequential_dense_1_matmul_readvariableop_resource:
??A
2sequential_dense_1_biasadd_readvariableop_resource:	?E
1sequential_dense_2_matmul_readvariableop_resource:
??A
2sequential_dense_2_biasadd_readvariableop_resource:	?D
1sequential_dense_3_matmul_readvariableop_resource:	?@
2sequential_dense_3_biasadd_readvariableop_resource:
identity??'sequential/dense/BiasAdd/ReadVariableOp?&sequential/dense/MatMul/ReadVariableOp?)sequential/dense_1/BiasAdd/ReadVariableOp?(sequential/dense_1/MatMul/ReadVariableOp?)sequential/dense_2/BiasAdd/ReadVariableOp?(sequential/dense_2/MatMul/ReadVariableOp?)sequential/dense_3/BiasAdd/ReadVariableOp?(sequential/dense_3/MatMul/ReadVariableOp?Tsequential/dense_features/Application_Area_1_indicator/None_Lookup/LookupTableFindV2?Tsequential/dense_features/Application_Area_2_indicator/None_Lookup/LookupTableFindV2?Osequential/dense_features/Cowork_Abroad_indicator/None_Lookup/LookupTableFindV2?Lsequential/dense_features/Cowork_Cor_indicator/None_Lookup/LookupTableFindV2?Msequential/dense_features/Cowork_Inst_indicator/None_Lookup/LookupTableFindV2?Lsequential/dense_features/Cowork_Uni_indicator/None_Lookup/LookupTableFindV2?Lsequential/dense_features/Cowork_etc_indicator/None_Lookup/LookupTableFindV2?Msequential/dense_features/Econ_Social_indicator/None_Lookup/LookupTableFindV2?Lsequential/dense_features/Green_Tech_indicator/None_Lookup/LookupTableFindV2?Lsequential/dense_features/Multi_Year_indicator/None_Lookup/LookupTableFindV2?Usequential/dense_features/National_Strategy_2_indicator/None_Lookup/LookupTableFindV2?Isequential/dense_features/RnD_Org_indicator/None_Lookup/LookupTableFindV2?Ksequential/dense_features/RnD_Stage_indicator/None_Lookup/LookupTableFindV2?Msequential/dense_features/STP_Code_11_indicator/None_Lookup/LookupTableFindV2?Msequential/dense_features/STP_Code_21_indicator/None_Lookup/LookupTableFindV2?Hsequential/dense_features/SixT_2_indicator/None_Lookup/LookupTableFindV2?
9sequential/dense_features/Application_Area_1_Weight/ShapeShapeapplication_area_1_weight*
T0*
_output_shapes
:?
Gsequential/dense_features/Application_Area_1_Weight/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
Isequential/dense_features/Application_Area_1_Weight/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
Isequential/dense_features/Application_Area_1_Weight/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
Asequential/dense_features/Application_Area_1_Weight/strided_sliceStridedSliceBsequential/dense_features/Application_Area_1_Weight/Shape:output:0Psequential/dense_features/Application_Area_1_Weight/strided_slice/stack:output:0Rsequential/dense_features/Application_Area_1_Weight/strided_slice/stack_1:output:0Rsequential/dense_features/Application_Area_1_Weight/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
Csequential/dense_features/Application_Area_1_Weight/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
Asequential/dense_features/Application_Area_1_Weight/Reshape/shapePackJsequential/dense_features/Application_Area_1_Weight/strided_slice:output:0Lsequential/dense_features/Application_Area_1_Weight/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
;sequential/dense_features/Application_Area_1_Weight/ReshapeReshapeapplication_area_1_weightJsequential/dense_features/Application_Area_1_Weight/Reshape/shape:output:0*
T0*'
_output_shapes
:??????????
Usequential/dense_features/Application_Area_1_indicator/to_sparse_input/ignore_value/xConst*
_output_shapes
: *
dtype0*
valueB B ?
Osequential/dense_features/Application_Area_1_indicator/to_sparse_input/NotEqualNotEqualapplication_area_1^sequential/dense_features/Application_Area_1_indicator/to_sparse_input/ignore_value/x:output:0*
T0*'
_output_shapes
:??????????
Nsequential/dense_features/Application_Area_1_indicator/to_sparse_input/indicesWhereSsequential/dense_features/Application_Area_1_indicator/to_sparse_input/NotEqual:z:0*'
_output_shapes
:??????????
Msequential/dense_features/Application_Area_1_indicator/to_sparse_input/valuesGatherNdapplication_area_1Vsequential/dense_features/Application_Area_1_indicator/to_sparse_input/indices:index:0*
Tindices0	*
Tparams0*#
_output_shapes
:??????????
Rsequential/dense_features/Application_Area_1_indicator/to_sparse_input/dense_shapeShapeapplication_area_1*
T0*
_output_shapes
:*
out_type0	?
Tsequential/dense_features/Application_Area_1_indicator/None_Lookup/LookupTableFindV2LookupTableFindV2asequential_dense_features_application_area_1_indicator_none_lookup_lookuptablefindv2_table_handleVsequential/dense_features/Application_Area_1_indicator/to_sparse_input/values:output:0bsequential_dense_features_application_area_1_indicator_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:??????????
Rsequential/dense_features/Application_Area_1_indicator/SparseToDense/default_valueConst*
_output_shapes
: *
dtype0	*
valueB	 R
??????????
Dsequential/dense_features/Application_Area_1_indicator/SparseToDenseSparseToDenseVsequential/dense_features/Application_Area_1_indicator/to_sparse_input/indices:index:0[sequential/dense_features/Application_Area_1_indicator/to_sparse_input/dense_shape:output:0]sequential/dense_features/Application_Area_1_indicator/None_Lookup/LookupTableFindV2:values:0[sequential/dense_features/Application_Area_1_indicator/SparseToDense/default_value:output:0*
T0	*
Tindices0	*'
_output_shapes
:??????????
Dsequential/dense_features/Application_Area_1_indicator/one_hot/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
Fsequential/dense_features/Application_Area_1_indicator/one_hot/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    ?
Dsequential/dense_features/Application_Area_1_indicator/one_hot/depthConst*
_output_shapes
: *
dtype0*
value	B :!?
>sequential/dense_features/Application_Area_1_indicator/one_hotOneHotLsequential/dense_features/Application_Area_1_indicator/SparseToDense:dense:0Msequential/dense_features/Application_Area_1_indicator/one_hot/depth:output:0Msequential/dense_features/Application_Area_1_indicator/one_hot/Const:output:0Osequential/dense_features/Application_Area_1_indicator/one_hot/Const_1:output:0*
T0*+
_output_shapes
:?????????!?
Lsequential/dense_features/Application_Area_1_indicator/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
??????????
:sequential/dense_features/Application_Area_1_indicator/SumSumGsequential/dense_features/Application_Area_1_indicator/one_hot:output:0Usequential/dense_features/Application_Area_1_indicator/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:?????????!?
<sequential/dense_features/Application_Area_1_indicator/ShapeShapeCsequential/dense_features/Application_Area_1_indicator/Sum:output:0*
T0*
_output_shapes
:?
Jsequential/dense_features/Application_Area_1_indicator/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
Lsequential/dense_features/Application_Area_1_indicator/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
Lsequential/dense_features/Application_Area_1_indicator/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
Dsequential/dense_features/Application_Area_1_indicator/strided_sliceStridedSliceEsequential/dense_features/Application_Area_1_indicator/Shape:output:0Ssequential/dense_features/Application_Area_1_indicator/strided_slice/stack:output:0Usequential/dense_features/Application_Area_1_indicator/strided_slice/stack_1:output:0Usequential/dense_features/Application_Area_1_indicator/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
Fsequential/dense_features/Application_Area_1_indicator/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :!?
Dsequential/dense_features/Application_Area_1_indicator/Reshape/shapePackMsequential/dense_features/Application_Area_1_indicator/strided_slice:output:0Osequential/dense_features/Application_Area_1_indicator/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
>sequential/dense_features/Application_Area_1_indicator/ReshapeReshapeCsequential/dense_features/Application_Area_1_indicator/Sum:output:0Msequential/dense_features/Application_Area_1_indicator/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????!?
9sequential/dense_features/Application_Area_2_Weight/ShapeShapeapplication_area_2_weight*
T0*
_output_shapes
:?
Gsequential/dense_features/Application_Area_2_Weight/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
Isequential/dense_features/Application_Area_2_Weight/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
Isequential/dense_features/Application_Area_2_Weight/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
Asequential/dense_features/Application_Area_2_Weight/strided_sliceStridedSliceBsequential/dense_features/Application_Area_2_Weight/Shape:output:0Psequential/dense_features/Application_Area_2_Weight/strided_slice/stack:output:0Rsequential/dense_features/Application_Area_2_Weight/strided_slice/stack_1:output:0Rsequential/dense_features/Application_Area_2_Weight/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
Csequential/dense_features/Application_Area_2_Weight/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
Asequential/dense_features/Application_Area_2_Weight/Reshape/shapePackJsequential/dense_features/Application_Area_2_Weight/strided_slice:output:0Lsequential/dense_features/Application_Area_2_Weight/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
;sequential/dense_features/Application_Area_2_Weight/ReshapeReshapeapplication_area_2_weightJsequential/dense_features/Application_Area_2_Weight/Reshape/shape:output:0*
T0*'
_output_shapes
:??????????
Usequential/dense_features/Application_Area_2_indicator/to_sparse_input/ignore_value/xConst*
_output_shapes
: *
dtype0*
valueB B ?
Osequential/dense_features/Application_Area_2_indicator/to_sparse_input/NotEqualNotEqualapplication_area_2^sequential/dense_features/Application_Area_2_indicator/to_sparse_input/ignore_value/x:output:0*
T0*'
_output_shapes
:??????????
Nsequential/dense_features/Application_Area_2_indicator/to_sparse_input/indicesWhereSsequential/dense_features/Application_Area_2_indicator/to_sparse_input/NotEqual:z:0*'
_output_shapes
:??????????
Msequential/dense_features/Application_Area_2_indicator/to_sparse_input/valuesGatherNdapplication_area_2Vsequential/dense_features/Application_Area_2_indicator/to_sparse_input/indices:index:0*
Tindices0	*
Tparams0*#
_output_shapes
:??????????
Rsequential/dense_features/Application_Area_2_indicator/to_sparse_input/dense_shapeShapeapplication_area_2*
T0*
_output_shapes
:*
out_type0	?
Tsequential/dense_features/Application_Area_2_indicator/None_Lookup/LookupTableFindV2LookupTableFindV2asequential_dense_features_application_area_2_indicator_none_lookup_lookuptablefindv2_table_handleVsequential/dense_features/Application_Area_2_indicator/to_sparse_input/values:output:0bsequential_dense_features_application_area_2_indicator_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:??????????
Rsequential/dense_features/Application_Area_2_indicator/SparseToDense/default_valueConst*
_output_shapes
: *
dtype0	*
valueB	 R
??????????
Dsequential/dense_features/Application_Area_2_indicator/SparseToDenseSparseToDenseVsequential/dense_features/Application_Area_2_indicator/to_sparse_input/indices:index:0[sequential/dense_features/Application_Area_2_indicator/to_sparse_input/dense_shape:output:0]sequential/dense_features/Application_Area_2_indicator/None_Lookup/LookupTableFindV2:values:0[sequential/dense_features/Application_Area_2_indicator/SparseToDense/default_value:output:0*
T0	*
Tindices0	*'
_output_shapes
:??????????
Dsequential/dense_features/Application_Area_2_indicator/one_hot/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
Fsequential/dense_features/Application_Area_2_indicator/one_hot/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    ?
Dsequential/dense_features/Application_Area_2_indicator/one_hot/depthConst*
_output_shapes
: *
dtype0*
value	B :"?
>sequential/dense_features/Application_Area_2_indicator/one_hotOneHotLsequential/dense_features/Application_Area_2_indicator/SparseToDense:dense:0Msequential/dense_features/Application_Area_2_indicator/one_hot/depth:output:0Msequential/dense_features/Application_Area_2_indicator/one_hot/Const:output:0Osequential/dense_features/Application_Area_2_indicator/one_hot/Const_1:output:0*
T0*+
_output_shapes
:?????????"?
Lsequential/dense_features/Application_Area_2_indicator/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
??????????
:sequential/dense_features/Application_Area_2_indicator/SumSumGsequential/dense_features/Application_Area_2_indicator/one_hot:output:0Usequential/dense_features/Application_Area_2_indicator/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:?????????"?
<sequential/dense_features/Application_Area_2_indicator/ShapeShapeCsequential/dense_features/Application_Area_2_indicator/Sum:output:0*
T0*
_output_shapes
:?
Jsequential/dense_features/Application_Area_2_indicator/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
Lsequential/dense_features/Application_Area_2_indicator/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
Lsequential/dense_features/Application_Area_2_indicator/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
Dsequential/dense_features/Application_Area_2_indicator/strided_sliceStridedSliceEsequential/dense_features/Application_Area_2_indicator/Shape:output:0Ssequential/dense_features/Application_Area_2_indicator/strided_slice/stack:output:0Usequential/dense_features/Application_Area_2_indicator/strided_slice/stack_1:output:0Usequential/dense_features/Application_Area_2_indicator/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
Fsequential/dense_features/Application_Area_2_indicator/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :"?
Dsequential/dense_features/Application_Area_2_indicator/Reshape/shapePackMsequential/dense_features/Application_Area_2_indicator/strided_slice:output:0Osequential/dense_features/Application_Area_2_indicator/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
>sequential/dense_features/Application_Area_2_indicator/ReshapeReshapeCsequential/dense_features/Application_Area_2_indicator/Sum:output:0Msequential/dense_features/Application_Area_2_indicator/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????"?
Psequential/dense_features/Cowork_Abroad_indicator/to_sparse_input/ignore_value/xConst*
_output_shapes
: *
dtype0*
valueB B ?
Jsequential/dense_features/Cowork_Abroad_indicator/to_sparse_input/NotEqualNotEqualcowork_abroadYsequential/dense_features/Cowork_Abroad_indicator/to_sparse_input/ignore_value/x:output:0*
T0*'
_output_shapes
:??????????
Isequential/dense_features/Cowork_Abroad_indicator/to_sparse_input/indicesWhereNsequential/dense_features/Cowork_Abroad_indicator/to_sparse_input/NotEqual:z:0*'
_output_shapes
:??????????
Hsequential/dense_features/Cowork_Abroad_indicator/to_sparse_input/valuesGatherNdcowork_abroadQsequential/dense_features/Cowork_Abroad_indicator/to_sparse_input/indices:index:0*
Tindices0	*
Tparams0*#
_output_shapes
:??????????
Msequential/dense_features/Cowork_Abroad_indicator/to_sparse_input/dense_shapeShapecowork_abroad*
T0*
_output_shapes
:*
out_type0	?
Osequential/dense_features/Cowork_Abroad_indicator/None_Lookup/LookupTableFindV2LookupTableFindV2\sequential_dense_features_cowork_abroad_indicator_none_lookup_lookuptablefindv2_table_handleQsequential/dense_features/Cowork_Abroad_indicator/to_sparse_input/values:output:0]sequential_dense_features_cowork_abroad_indicator_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:??????????
Msequential/dense_features/Cowork_Abroad_indicator/SparseToDense/default_valueConst*
_output_shapes
: *
dtype0	*
valueB	 R
??????????
?sequential/dense_features/Cowork_Abroad_indicator/SparseToDenseSparseToDenseQsequential/dense_features/Cowork_Abroad_indicator/to_sparse_input/indices:index:0Vsequential/dense_features/Cowork_Abroad_indicator/to_sparse_input/dense_shape:output:0Xsequential/dense_features/Cowork_Abroad_indicator/None_Lookup/LookupTableFindV2:values:0Vsequential/dense_features/Cowork_Abroad_indicator/SparseToDense/default_value:output:0*
T0	*
Tindices0	*'
_output_shapes
:??????????
?sequential/dense_features/Cowork_Abroad_indicator/one_hot/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
Asequential/dense_features/Cowork_Abroad_indicator/one_hot/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    ?
?sequential/dense_features/Cowork_Abroad_indicator/one_hot/depthConst*
_output_shapes
: *
dtype0*
value	B :?
9sequential/dense_features/Cowork_Abroad_indicator/one_hotOneHotGsequential/dense_features/Cowork_Abroad_indicator/SparseToDense:dense:0Hsequential/dense_features/Cowork_Abroad_indicator/one_hot/depth:output:0Hsequential/dense_features/Cowork_Abroad_indicator/one_hot/Const:output:0Jsequential/dense_features/Cowork_Abroad_indicator/one_hot/Const_1:output:0*
T0*+
_output_shapes
:??????????
Gsequential/dense_features/Cowork_Abroad_indicator/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
??????????
5sequential/dense_features/Cowork_Abroad_indicator/SumSumBsequential/dense_features/Cowork_Abroad_indicator/one_hot:output:0Psequential/dense_features/Cowork_Abroad_indicator/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:??????????
7sequential/dense_features/Cowork_Abroad_indicator/ShapeShape>sequential/dense_features/Cowork_Abroad_indicator/Sum:output:0*
T0*
_output_shapes
:?
Esequential/dense_features/Cowork_Abroad_indicator/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
Gsequential/dense_features/Cowork_Abroad_indicator/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
Gsequential/dense_features/Cowork_Abroad_indicator/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
?sequential/dense_features/Cowork_Abroad_indicator/strided_sliceStridedSlice@sequential/dense_features/Cowork_Abroad_indicator/Shape:output:0Nsequential/dense_features/Cowork_Abroad_indicator/strided_slice/stack:output:0Psequential/dense_features/Cowork_Abroad_indicator/strided_slice/stack_1:output:0Psequential/dense_features/Cowork_Abroad_indicator/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
Asequential/dense_features/Cowork_Abroad_indicator/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
?sequential/dense_features/Cowork_Abroad_indicator/Reshape/shapePackHsequential/dense_features/Cowork_Abroad_indicator/strided_slice:output:0Jsequential/dense_features/Cowork_Abroad_indicator/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
9sequential/dense_features/Cowork_Abroad_indicator/ReshapeReshape>sequential/dense_features/Cowork_Abroad_indicator/Sum:output:0Hsequential/dense_features/Cowork_Abroad_indicator/Reshape/shape:output:0*
T0*'
_output_shapes
:??????????
Msequential/dense_features/Cowork_Cor_indicator/to_sparse_input/ignore_value/xConst*
_output_shapes
: *
dtype0*
valueB B ?
Gsequential/dense_features/Cowork_Cor_indicator/to_sparse_input/NotEqualNotEqual
cowork_corVsequential/dense_features/Cowork_Cor_indicator/to_sparse_input/ignore_value/x:output:0*
T0*'
_output_shapes
:??????????
Fsequential/dense_features/Cowork_Cor_indicator/to_sparse_input/indicesWhereKsequential/dense_features/Cowork_Cor_indicator/to_sparse_input/NotEqual:z:0*'
_output_shapes
:??????????
Esequential/dense_features/Cowork_Cor_indicator/to_sparse_input/valuesGatherNd
cowork_corNsequential/dense_features/Cowork_Cor_indicator/to_sparse_input/indices:index:0*
Tindices0	*
Tparams0*#
_output_shapes
:??????????
Jsequential/dense_features/Cowork_Cor_indicator/to_sparse_input/dense_shapeShape
cowork_cor*
T0*
_output_shapes
:*
out_type0	?
Lsequential/dense_features/Cowork_Cor_indicator/None_Lookup/LookupTableFindV2LookupTableFindV2Ysequential_dense_features_cowork_cor_indicator_none_lookup_lookuptablefindv2_table_handleNsequential/dense_features/Cowork_Cor_indicator/to_sparse_input/values:output:0Zsequential_dense_features_cowork_cor_indicator_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:??????????
Jsequential/dense_features/Cowork_Cor_indicator/SparseToDense/default_valueConst*
_output_shapes
: *
dtype0	*
valueB	 R
??????????
<sequential/dense_features/Cowork_Cor_indicator/SparseToDenseSparseToDenseNsequential/dense_features/Cowork_Cor_indicator/to_sparse_input/indices:index:0Ssequential/dense_features/Cowork_Cor_indicator/to_sparse_input/dense_shape:output:0Usequential/dense_features/Cowork_Cor_indicator/None_Lookup/LookupTableFindV2:values:0Ssequential/dense_features/Cowork_Cor_indicator/SparseToDense/default_value:output:0*
T0	*
Tindices0	*'
_output_shapes
:??????????
<sequential/dense_features/Cowork_Cor_indicator/one_hot/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
>sequential/dense_features/Cowork_Cor_indicator/one_hot/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    ~
<sequential/dense_features/Cowork_Cor_indicator/one_hot/depthConst*
_output_shapes
: *
dtype0*
value	B :?
6sequential/dense_features/Cowork_Cor_indicator/one_hotOneHotDsequential/dense_features/Cowork_Cor_indicator/SparseToDense:dense:0Esequential/dense_features/Cowork_Cor_indicator/one_hot/depth:output:0Esequential/dense_features/Cowork_Cor_indicator/one_hot/Const:output:0Gsequential/dense_features/Cowork_Cor_indicator/one_hot/Const_1:output:0*
T0*+
_output_shapes
:??????????
Dsequential/dense_features/Cowork_Cor_indicator/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
??????????
2sequential/dense_features/Cowork_Cor_indicator/SumSum?sequential/dense_features/Cowork_Cor_indicator/one_hot:output:0Msequential/dense_features/Cowork_Cor_indicator/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:??????????
4sequential/dense_features/Cowork_Cor_indicator/ShapeShape;sequential/dense_features/Cowork_Cor_indicator/Sum:output:0*
T0*
_output_shapes
:?
Bsequential/dense_features/Cowork_Cor_indicator/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
Dsequential/dense_features/Cowork_Cor_indicator/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
Dsequential/dense_features/Cowork_Cor_indicator/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
<sequential/dense_features/Cowork_Cor_indicator/strided_sliceStridedSlice=sequential/dense_features/Cowork_Cor_indicator/Shape:output:0Ksequential/dense_features/Cowork_Cor_indicator/strided_slice/stack:output:0Msequential/dense_features/Cowork_Cor_indicator/strided_slice/stack_1:output:0Msequential/dense_features/Cowork_Cor_indicator/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
>sequential/dense_features/Cowork_Cor_indicator/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
<sequential/dense_features/Cowork_Cor_indicator/Reshape/shapePackEsequential/dense_features/Cowork_Cor_indicator/strided_slice:output:0Gsequential/dense_features/Cowork_Cor_indicator/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
6sequential/dense_features/Cowork_Cor_indicator/ReshapeReshape;sequential/dense_features/Cowork_Cor_indicator/Sum:output:0Esequential/dense_features/Cowork_Cor_indicator/Reshape/shape:output:0*
T0*'
_output_shapes
:??????????
Nsequential/dense_features/Cowork_Inst_indicator/to_sparse_input/ignore_value/xConst*
_output_shapes
: *
dtype0*
valueB B ?
Hsequential/dense_features/Cowork_Inst_indicator/to_sparse_input/NotEqualNotEqualcowork_instWsequential/dense_features/Cowork_Inst_indicator/to_sparse_input/ignore_value/x:output:0*
T0*'
_output_shapes
:??????????
Gsequential/dense_features/Cowork_Inst_indicator/to_sparse_input/indicesWhereLsequential/dense_features/Cowork_Inst_indicator/to_sparse_input/NotEqual:z:0*'
_output_shapes
:??????????
Fsequential/dense_features/Cowork_Inst_indicator/to_sparse_input/valuesGatherNdcowork_instOsequential/dense_features/Cowork_Inst_indicator/to_sparse_input/indices:index:0*
Tindices0	*
Tparams0*#
_output_shapes
:??????????
Ksequential/dense_features/Cowork_Inst_indicator/to_sparse_input/dense_shapeShapecowork_inst*
T0*
_output_shapes
:*
out_type0	?
Msequential/dense_features/Cowork_Inst_indicator/None_Lookup/LookupTableFindV2LookupTableFindV2Zsequential_dense_features_cowork_inst_indicator_none_lookup_lookuptablefindv2_table_handleOsequential/dense_features/Cowork_Inst_indicator/to_sparse_input/values:output:0[sequential_dense_features_cowork_inst_indicator_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:??????????
Ksequential/dense_features/Cowork_Inst_indicator/SparseToDense/default_valueConst*
_output_shapes
: *
dtype0	*
valueB	 R
??????????
=sequential/dense_features/Cowork_Inst_indicator/SparseToDenseSparseToDenseOsequential/dense_features/Cowork_Inst_indicator/to_sparse_input/indices:index:0Tsequential/dense_features/Cowork_Inst_indicator/to_sparse_input/dense_shape:output:0Vsequential/dense_features/Cowork_Inst_indicator/None_Lookup/LookupTableFindV2:values:0Tsequential/dense_features/Cowork_Inst_indicator/SparseToDense/default_value:output:0*
T0	*
Tindices0	*'
_output_shapes
:??????????
=sequential/dense_features/Cowork_Inst_indicator/one_hot/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
?sequential/dense_features/Cowork_Inst_indicator/one_hot/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    
=sequential/dense_features/Cowork_Inst_indicator/one_hot/depthConst*
_output_shapes
: *
dtype0*
value	B :?
7sequential/dense_features/Cowork_Inst_indicator/one_hotOneHotEsequential/dense_features/Cowork_Inst_indicator/SparseToDense:dense:0Fsequential/dense_features/Cowork_Inst_indicator/one_hot/depth:output:0Fsequential/dense_features/Cowork_Inst_indicator/one_hot/Const:output:0Hsequential/dense_features/Cowork_Inst_indicator/one_hot/Const_1:output:0*
T0*+
_output_shapes
:??????????
Esequential/dense_features/Cowork_Inst_indicator/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
??????????
3sequential/dense_features/Cowork_Inst_indicator/SumSum@sequential/dense_features/Cowork_Inst_indicator/one_hot:output:0Nsequential/dense_features/Cowork_Inst_indicator/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:??????????
5sequential/dense_features/Cowork_Inst_indicator/ShapeShape<sequential/dense_features/Cowork_Inst_indicator/Sum:output:0*
T0*
_output_shapes
:?
Csequential/dense_features/Cowork_Inst_indicator/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
Esequential/dense_features/Cowork_Inst_indicator/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
Esequential/dense_features/Cowork_Inst_indicator/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
=sequential/dense_features/Cowork_Inst_indicator/strided_sliceStridedSlice>sequential/dense_features/Cowork_Inst_indicator/Shape:output:0Lsequential/dense_features/Cowork_Inst_indicator/strided_slice/stack:output:0Nsequential/dense_features/Cowork_Inst_indicator/strided_slice/stack_1:output:0Nsequential/dense_features/Cowork_Inst_indicator/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
?sequential/dense_features/Cowork_Inst_indicator/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
=sequential/dense_features/Cowork_Inst_indicator/Reshape/shapePackFsequential/dense_features/Cowork_Inst_indicator/strided_slice:output:0Hsequential/dense_features/Cowork_Inst_indicator/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
7sequential/dense_features/Cowork_Inst_indicator/ReshapeReshape<sequential/dense_features/Cowork_Inst_indicator/Sum:output:0Fsequential/dense_features/Cowork_Inst_indicator/Reshape/shape:output:0*
T0*'
_output_shapes
:??????????
Msequential/dense_features/Cowork_Uni_indicator/to_sparse_input/ignore_value/xConst*
_output_shapes
: *
dtype0*
valueB B ?
Gsequential/dense_features/Cowork_Uni_indicator/to_sparse_input/NotEqualNotEqual
cowork_uniVsequential/dense_features/Cowork_Uni_indicator/to_sparse_input/ignore_value/x:output:0*
T0*'
_output_shapes
:??????????
Fsequential/dense_features/Cowork_Uni_indicator/to_sparse_input/indicesWhereKsequential/dense_features/Cowork_Uni_indicator/to_sparse_input/NotEqual:z:0*'
_output_shapes
:??????????
Esequential/dense_features/Cowork_Uni_indicator/to_sparse_input/valuesGatherNd
cowork_uniNsequential/dense_features/Cowork_Uni_indicator/to_sparse_input/indices:index:0*
Tindices0	*
Tparams0*#
_output_shapes
:??????????
Jsequential/dense_features/Cowork_Uni_indicator/to_sparse_input/dense_shapeShape
cowork_uni*
T0*
_output_shapes
:*
out_type0	?
Lsequential/dense_features/Cowork_Uni_indicator/None_Lookup/LookupTableFindV2LookupTableFindV2Ysequential_dense_features_cowork_uni_indicator_none_lookup_lookuptablefindv2_table_handleNsequential/dense_features/Cowork_Uni_indicator/to_sparse_input/values:output:0Zsequential_dense_features_cowork_uni_indicator_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:??????????
Jsequential/dense_features/Cowork_Uni_indicator/SparseToDense/default_valueConst*
_output_shapes
: *
dtype0	*
valueB	 R
??????????
<sequential/dense_features/Cowork_Uni_indicator/SparseToDenseSparseToDenseNsequential/dense_features/Cowork_Uni_indicator/to_sparse_input/indices:index:0Ssequential/dense_features/Cowork_Uni_indicator/to_sparse_input/dense_shape:output:0Usequential/dense_features/Cowork_Uni_indicator/None_Lookup/LookupTableFindV2:values:0Ssequential/dense_features/Cowork_Uni_indicator/SparseToDense/default_value:output:0*
T0	*
Tindices0	*'
_output_shapes
:??????????
<sequential/dense_features/Cowork_Uni_indicator/one_hot/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
>sequential/dense_features/Cowork_Uni_indicator/one_hot/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    ~
<sequential/dense_features/Cowork_Uni_indicator/one_hot/depthConst*
_output_shapes
: *
dtype0*
value	B :?
6sequential/dense_features/Cowork_Uni_indicator/one_hotOneHotDsequential/dense_features/Cowork_Uni_indicator/SparseToDense:dense:0Esequential/dense_features/Cowork_Uni_indicator/one_hot/depth:output:0Esequential/dense_features/Cowork_Uni_indicator/one_hot/Const:output:0Gsequential/dense_features/Cowork_Uni_indicator/one_hot/Const_1:output:0*
T0*+
_output_shapes
:??????????
Dsequential/dense_features/Cowork_Uni_indicator/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
??????????
2sequential/dense_features/Cowork_Uni_indicator/SumSum?sequential/dense_features/Cowork_Uni_indicator/one_hot:output:0Msequential/dense_features/Cowork_Uni_indicator/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:??????????
4sequential/dense_features/Cowork_Uni_indicator/ShapeShape;sequential/dense_features/Cowork_Uni_indicator/Sum:output:0*
T0*
_output_shapes
:?
Bsequential/dense_features/Cowork_Uni_indicator/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
Dsequential/dense_features/Cowork_Uni_indicator/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
Dsequential/dense_features/Cowork_Uni_indicator/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
<sequential/dense_features/Cowork_Uni_indicator/strided_sliceStridedSlice=sequential/dense_features/Cowork_Uni_indicator/Shape:output:0Ksequential/dense_features/Cowork_Uni_indicator/strided_slice/stack:output:0Msequential/dense_features/Cowork_Uni_indicator/strided_slice/stack_1:output:0Msequential/dense_features/Cowork_Uni_indicator/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
>sequential/dense_features/Cowork_Uni_indicator/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
<sequential/dense_features/Cowork_Uni_indicator/Reshape/shapePackEsequential/dense_features/Cowork_Uni_indicator/strided_slice:output:0Gsequential/dense_features/Cowork_Uni_indicator/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
6sequential/dense_features/Cowork_Uni_indicator/ReshapeReshape;sequential/dense_features/Cowork_Uni_indicator/Sum:output:0Esequential/dense_features/Cowork_Uni_indicator/Reshape/shape:output:0*
T0*'
_output_shapes
:??????????
Msequential/dense_features/Cowork_etc_indicator/to_sparse_input/ignore_value/xConst*
_output_shapes
: *
dtype0*
valueB B ?
Gsequential/dense_features/Cowork_etc_indicator/to_sparse_input/NotEqualNotEqual
cowork_etcVsequential/dense_features/Cowork_etc_indicator/to_sparse_input/ignore_value/x:output:0*
T0*'
_output_shapes
:??????????
Fsequential/dense_features/Cowork_etc_indicator/to_sparse_input/indicesWhereKsequential/dense_features/Cowork_etc_indicator/to_sparse_input/NotEqual:z:0*'
_output_shapes
:??????????
Esequential/dense_features/Cowork_etc_indicator/to_sparse_input/valuesGatherNd
cowork_etcNsequential/dense_features/Cowork_etc_indicator/to_sparse_input/indices:index:0*
Tindices0	*
Tparams0*#
_output_shapes
:??????????
Jsequential/dense_features/Cowork_etc_indicator/to_sparse_input/dense_shapeShape
cowork_etc*
T0*
_output_shapes
:*
out_type0	?
Lsequential/dense_features/Cowork_etc_indicator/None_Lookup/LookupTableFindV2LookupTableFindV2Ysequential_dense_features_cowork_etc_indicator_none_lookup_lookuptablefindv2_table_handleNsequential/dense_features/Cowork_etc_indicator/to_sparse_input/values:output:0Zsequential_dense_features_cowork_etc_indicator_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:??????????
Jsequential/dense_features/Cowork_etc_indicator/SparseToDense/default_valueConst*
_output_shapes
: *
dtype0	*
valueB	 R
??????????
<sequential/dense_features/Cowork_etc_indicator/SparseToDenseSparseToDenseNsequential/dense_features/Cowork_etc_indicator/to_sparse_input/indices:index:0Ssequential/dense_features/Cowork_etc_indicator/to_sparse_input/dense_shape:output:0Usequential/dense_features/Cowork_etc_indicator/None_Lookup/LookupTableFindV2:values:0Ssequential/dense_features/Cowork_etc_indicator/SparseToDense/default_value:output:0*
T0	*
Tindices0	*'
_output_shapes
:??????????
<sequential/dense_features/Cowork_etc_indicator/one_hot/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
>sequential/dense_features/Cowork_etc_indicator/one_hot/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    ~
<sequential/dense_features/Cowork_etc_indicator/one_hot/depthConst*
_output_shapes
: *
dtype0*
value	B :?
6sequential/dense_features/Cowork_etc_indicator/one_hotOneHotDsequential/dense_features/Cowork_etc_indicator/SparseToDense:dense:0Esequential/dense_features/Cowork_etc_indicator/one_hot/depth:output:0Esequential/dense_features/Cowork_etc_indicator/one_hot/Const:output:0Gsequential/dense_features/Cowork_etc_indicator/one_hot/Const_1:output:0*
T0*+
_output_shapes
:??????????
Dsequential/dense_features/Cowork_etc_indicator/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
??????????
2sequential/dense_features/Cowork_etc_indicator/SumSum?sequential/dense_features/Cowork_etc_indicator/one_hot:output:0Msequential/dense_features/Cowork_etc_indicator/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:??????????
4sequential/dense_features/Cowork_etc_indicator/ShapeShape;sequential/dense_features/Cowork_etc_indicator/Sum:output:0*
T0*
_output_shapes
:?
Bsequential/dense_features/Cowork_etc_indicator/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
Dsequential/dense_features/Cowork_etc_indicator/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
Dsequential/dense_features/Cowork_etc_indicator/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
<sequential/dense_features/Cowork_etc_indicator/strided_sliceStridedSlice=sequential/dense_features/Cowork_etc_indicator/Shape:output:0Ksequential/dense_features/Cowork_etc_indicator/strided_slice/stack:output:0Msequential/dense_features/Cowork_etc_indicator/strided_slice/stack_1:output:0Msequential/dense_features/Cowork_etc_indicator/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
>sequential/dense_features/Cowork_etc_indicator/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
<sequential/dense_features/Cowork_etc_indicator/Reshape/shapePackEsequential/dense_features/Cowork_etc_indicator/strided_slice:output:0Gsequential/dense_features/Cowork_etc_indicator/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
6sequential/dense_features/Cowork_etc_indicator/ReshapeReshape;sequential/dense_features/Cowork_etc_indicator/Sum:output:0Esequential/dense_features/Cowork_etc_indicator/Reshape/shape:output:0*
T0*'
_output_shapes
:??????????
Nsequential/dense_features/Econ_Social_indicator/to_sparse_input/ignore_value/xConst*
_output_shapes
: *
dtype0*
valueB :
??????????
Lsequential/dense_features/Econ_Social_indicator/to_sparse_input/ignore_valueCastWsequential/dense_features/Econ_Social_indicator/to_sparse_input/ignore_value/x:output:0*

DstT0	*

SrcT0*
_output_shapes
: ?
Hsequential/dense_features/Econ_Social_indicator/to_sparse_input/NotEqualNotEqualecon_socialPsequential/dense_features/Econ_Social_indicator/to_sparse_input/ignore_value:y:0*
T0	*'
_output_shapes
:??????????
Gsequential/dense_features/Econ_Social_indicator/to_sparse_input/indicesWhereLsequential/dense_features/Econ_Social_indicator/to_sparse_input/NotEqual:z:0*'
_output_shapes
:??????????
Fsequential/dense_features/Econ_Social_indicator/to_sparse_input/valuesGatherNdecon_socialOsequential/dense_features/Econ_Social_indicator/to_sparse_input/indices:index:0*
Tindices0	*
Tparams0	*#
_output_shapes
:??????????
Ksequential/dense_features/Econ_Social_indicator/to_sparse_input/dense_shapeShapeecon_social*
T0	*
_output_shapes
:*
out_type0	?
Msequential/dense_features/Econ_Social_indicator/None_Lookup/LookupTableFindV2LookupTableFindV2Zsequential_dense_features_econ_social_indicator_none_lookup_lookuptablefindv2_table_handleOsequential/dense_features/Econ_Social_indicator/to_sparse_input/values:output:0[sequential_dense_features_econ_social_indicator_none_lookup_lookuptablefindv2_default_value*	
Tin0	*

Tout0	*#
_output_shapes
:??????????
Ksequential/dense_features/Econ_Social_indicator/SparseToDense/default_valueConst*
_output_shapes
: *
dtype0	*
valueB	 R
??????????
=sequential/dense_features/Econ_Social_indicator/SparseToDenseSparseToDenseOsequential/dense_features/Econ_Social_indicator/to_sparse_input/indices:index:0Tsequential/dense_features/Econ_Social_indicator/to_sparse_input/dense_shape:output:0Vsequential/dense_features/Econ_Social_indicator/None_Lookup/LookupTableFindV2:values:0Tsequential/dense_features/Econ_Social_indicator/SparseToDense/default_value:output:0*
T0	*
Tindices0	*'
_output_shapes
:??????????
=sequential/dense_features/Econ_Social_indicator/one_hot/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
?sequential/dense_features/Econ_Social_indicator/one_hot/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    
=sequential/dense_features/Econ_Social_indicator/one_hot/depthConst*
_output_shapes
: *
dtype0*
value	B :?
7sequential/dense_features/Econ_Social_indicator/one_hotOneHotEsequential/dense_features/Econ_Social_indicator/SparseToDense:dense:0Fsequential/dense_features/Econ_Social_indicator/one_hot/depth:output:0Fsequential/dense_features/Econ_Social_indicator/one_hot/Const:output:0Hsequential/dense_features/Econ_Social_indicator/one_hot/Const_1:output:0*
T0*+
_output_shapes
:??????????
Esequential/dense_features/Econ_Social_indicator/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
??????????
3sequential/dense_features/Econ_Social_indicator/SumSum@sequential/dense_features/Econ_Social_indicator/one_hot:output:0Nsequential/dense_features/Econ_Social_indicator/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:??????????
5sequential/dense_features/Econ_Social_indicator/ShapeShape<sequential/dense_features/Econ_Social_indicator/Sum:output:0*
T0*
_output_shapes
:?
Csequential/dense_features/Econ_Social_indicator/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
Esequential/dense_features/Econ_Social_indicator/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
Esequential/dense_features/Econ_Social_indicator/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
=sequential/dense_features/Econ_Social_indicator/strided_sliceStridedSlice>sequential/dense_features/Econ_Social_indicator/Shape:output:0Lsequential/dense_features/Econ_Social_indicator/strided_slice/stack:output:0Nsequential/dense_features/Econ_Social_indicator/strided_slice/stack_1:output:0Nsequential/dense_features/Econ_Social_indicator/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
?sequential/dense_features/Econ_Social_indicator/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
=sequential/dense_features/Econ_Social_indicator/Reshape/shapePackFsequential/dense_features/Econ_Social_indicator/strided_slice:output:0Hsequential/dense_features/Econ_Social_indicator/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
7sequential/dense_features/Econ_Social_indicator/ReshapeReshape<sequential/dense_features/Econ_Social_indicator/Sum:output:0Fsequential/dense_features/Econ_Social_indicator/Reshape/shape:output:0*
T0*'
_output_shapes
:??????????
Msequential/dense_features/Green_Tech_indicator/to_sparse_input/ignore_value/xConst*
_output_shapes
: *
dtype0*
valueB :
??????????
Ksequential/dense_features/Green_Tech_indicator/to_sparse_input/ignore_valueCastVsequential/dense_features/Green_Tech_indicator/to_sparse_input/ignore_value/x:output:0*

DstT0	*

SrcT0*
_output_shapes
: ?
Gsequential/dense_features/Green_Tech_indicator/to_sparse_input/NotEqualNotEqual
green_techOsequential/dense_features/Green_Tech_indicator/to_sparse_input/ignore_value:y:0*
T0	*'
_output_shapes
:??????????
Fsequential/dense_features/Green_Tech_indicator/to_sparse_input/indicesWhereKsequential/dense_features/Green_Tech_indicator/to_sparse_input/NotEqual:z:0*'
_output_shapes
:??????????
Esequential/dense_features/Green_Tech_indicator/to_sparse_input/valuesGatherNd
green_techNsequential/dense_features/Green_Tech_indicator/to_sparse_input/indices:index:0*
Tindices0	*
Tparams0	*#
_output_shapes
:??????????
Jsequential/dense_features/Green_Tech_indicator/to_sparse_input/dense_shapeShape
green_tech*
T0	*
_output_shapes
:*
out_type0	?
Lsequential/dense_features/Green_Tech_indicator/None_Lookup/LookupTableFindV2LookupTableFindV2Ysequential_dense_features_green_tech_indicator_none_lookup_lookuptablefindv2_table_handleNsequential/dense_features/Green_Tech_indicator/to_sparse_input/values:output:0Zsequential_dense_features_green_tech_indicator_none_lookup_lookuptablefindv2_default_value*	
Tin0	*

Tout0	*#
_output_shapes
:??????????
Jsequential/dense_features/Green_Tech_indicator/SparseToDense/default_valueConst*
_output_shapes
: *
dtype0	*
valueB	 R
??????????
<sequential/dense_features/Green_Tech_indicator/SparseToDenseSparseToDenseNsequential/dense_features/Green_Tech_indicator/to_sparse_input/indices:index:0Ssequential/dense_features/Green_Tech_indicator/to_sparse_input/dense_shape:output:0Usequential/dense_features/Green_Tech_indicator/None_Lookup/LookupTableFindV2:values:0Ssequential/dense_features/Green_Tech_indicator/SparseToDense/default_value:output:0*
T0	*
Tindices0	*'
_output_shapes
:??????????
<sequential/dense_features/Green_Tech_indicator/one_hot/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
>sequential/dense_features/Green_Tech_indicator/one_hot/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    ~
<sequential/dense_features/Green_Tech_indicator/one_hot/depthConst*
_output_shapes
: *
dtype0*
value	B :<?
6sequential/dense_features/Green_Tech_indicator/one_hotOneHotDsequential/dense_features/Green_Tech_indicator/SparseToDense:dense:0Esequential/dense_features/Green_Tech_indicator/one_hot/depth:output:0Esequential/dense_features/Green_Tech_indicator/one_hot/Const:output:0Gsequential/dense_features/Green_Tech_indicator/one_hot/Const_1:output:0*
T0*+
_output_shapes
:?????????<?
Dsequential/dense_features/Green_Tech_indicator/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
??????????
2sequential/dense_features/Green_Tech_indicator/SumSum?sequential/dense_features/Green_Tech_indicator/one_hot:output:0Msequential/dense_features/Green_Tech_indicator/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:?????????<?
4sequential/dense_features/Green_Tech_indicator/ShapeShape;sequential/dense_features/Green_Tech_indicator/Sum:output:0*
T0*
_output_shapes
:?
Bsequential/dense_features/Green_Tech_indicator/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
Dsequential/dense_features/Green_Tech_indicator/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
Dsequential/dense_features/Green_Tech_indicator/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
<sequential/dense_features/Green_Tech_indicator/strided_sliceStridedSlice=sequential/dense_features/Green_Tech_indicator/Shape:output:0Ksequential/dense_features/Green_Tech_indicator/strided_slice/stack:output:0Msequential/dense_features/Green_Tech_indicator/strided_slice/stack_1:output:0Msequential/dense_features/Green_Tech_indicator/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
>sequential/dense_features/Green_Tech_indicator/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :<?
<sequential/dense_features/Green_Tech_indicator/Reshape/shapePackEsequential/dense_features/Green_Tech_indicator/strided_slice:output:0Gsequential/dense_features/Green_Tech_indicator/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
6sequential/dense_features/Green_Tech_indicator/ReshapeReshape;sequential/dense_features/Green_Tech_indicator/Sum:output:0Esequential/dense_features/Green_Tech_indicator/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????<h
,sequential/dense_features/Log_Duration/ShapeShapelog_duration*
T0*
_output_shapes
:?
:sequential/dense_features/Log_Duration/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
<sequential/dense_features/Log_Duration/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
<sequential/dense_features/Log_Duration/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
4sequential/dense_features/Log_Duration/strided_sliceStridedSlice5sequential/dense_features/Log_Duration/Shape:output:0Csequential/dense_features/Log_Duration/strided_slice/stack:output:0Esequential/dense_features/Log_Duration/strided_slice/stack_1:output:0Esequential/dense_features/Log_Duration/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskx
6sequential/dense_features/Log_Duration/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
4sequential/dense_features/Log_Duration/Reshape/shapePack=sequential/dense_features/Log_Duration/strided_slice:output:0?sequential/dense_features/Log_Duration/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
.sequential/dense_features/Log_Duration/ReshapeReshapelog_duration=sequential/dense_features/Log_Duration/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????h
,sequential/dense_features/Log_RnD_Fund/ShapeShapelog_rnd_fund*
T0*
_output_shapes
:?
:sequential/dense_features/Log_RnD_Fund/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
<sequential/dense_features/Log_RnD_Fund/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
<sequential/dense_features/Log_RnD_Fund/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
4sequential/dense_features/Log_RnD_Fund/strided_sliceStridedSlice5sequential/dense_features/Log_RnD_Fund/Shape:output:0Csequential/dense_features/Log_RnD_Fund/strided_slice/stack:output:0Esequential/dense_features/Log_RnD_Fund/strided_slice/stack_1:output:0Esequential/dense_features/Log_RnD_Fund/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskx
6sequential/dense_features/Log_RnD_Fund/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
4sequential/dense_features/Log_RnD_Fund/Reshape/shapePack=sequential/dense_features/Log_RnD_Fund/strided_slice:output:0?sequential/dense_features/Log_RnD_Fund/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
.sequential/dense_features/Log_RnD_Fund/ReshapeReshapelog_rnd_fund=sequential/dense_features/Log_RnD_Fund/Reshape/shape:output:0*
T0*'
_output_shapes
:??????????
Msequential/dense_features/Multi_Year_indicator/to_sparse_input/ignore_value/xConst*
_output_shapes
: *
dtype0*
valueB :
??????????
Ksequential/dense_features/Multi_Year_indicator/to_sparse_input/ignore_valueCastVsequential/dense_features/Multi_Year_indicator/to_sparse_input/ignore_value/x:output:0*

DstT0	*

SrcT0*
_output_shapes
: ?
Gsequential/dense_features/Multi_Year_indicator/to_sparse_input/NotEqualNotEqual
multi_yearOsequential/dense_features/Multi_Year_indicator/to_sparse_input/ignore_value:y:0*
T0	*'
_output_shapes
:??????????
Fsequential/dense_features/Multi_Year_indicator/to_sparse_input/indicesWhereKsequential/dense_features/Multi_Year_indicator/to_sparse_input/NotEqual:z:0*'
_output_shapes
:??????????
Esequential/dense_features/Multi_Year_indicator/to_sparse_input/valuesGatherNd
multi_yearNsequential/dense_features/Multi_Year_indicator/to_sparse_input/indices:index:0*
Tindices0	*
Tparams0	*#
_output_shapes
:??????????
Jsequential/dense_features/Multi_Year_indicator/to_sparse_input/dense_shapeShape
multi_year*
T0	*
_output_shapes
:*
out_type0	?
Lsequential/dense_features/Multi_Year_indicator/None_Lookup/LookupTableFindV2LookupTableFindV2Ysequential_dense_features_multi_year_indicator_none_lookup_lookuptablefindv2_table_handleNsequential/dense_features/Multi_Year_indicator/to_sparse_input/values:output:0Zsequential_dense_features_multi_year_indicator_none_lookup_lookuptablefindv2_default_value*	
Tin0	*

Tout0	*#
_output_shapes
:??????????
Jsequential/dense_features/Multi_Year_indicator/SparseToDense/default_valueConst*
_output_shapes
: *
dtype0	*
valueB	 R
??????????
<sequential/dense_features/Multi_Year_indicator/SparseToDenseSparseToDenseNsequential/dense_features/Multi_Year_indicator/to_sparse_input/indices:index:0Ssequential/dense_features/Multi_Year_indicator/to_sparse_input/dense_shape:output:0Usequential/dense_features/Multi_Year_indicator/None_Lookup/LookupTableFindV2:values:0Ssequential/dense_features/Multi_Year_indicator/SparseToDense/default_value:output:0*
T0	*
Tindices0	*'
_output_shapes
:??????????
<sequential/dense_features/Multi_Year_indicator/one_hot/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
>sequential/dense_features/Multi_Year_indicator/one_hot/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    ~
<sequential/dense_features/Multi_Year_indicator/one_hot/depthConst*
_output_shapes
: *
dtype0*
value	B :?
6sequential/dense_features/Multi_Year_indicator/one_hotOneHotDsequential/dense_features/Multi_Year_indicator/SparseToDense:dense:0Esequential/dense_features/Multi_Year_indicator/one_hot/depth:output:0Esequential/dense_features/Multi_Year_indicator/one_hot/Const:output:0Gsequential/dense_features/Multi_Year_indicator/one_hot/Const_1:output:0*
T0*+
_output_shapes
:??????????
Dsequential/dense_features/Multi_Year_indicator/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
??????????
2sequential/dense_features/Multi_Year_indicator/SumSum?sequential/dense_features/Multi_Year_indicator/one_hot:output:0Msequential/dense_features/Multi_Year_indicator/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:??????????
4sequential/dense_features/Multi_Year_indicator/ShapeShape;sequential/dense_features/Multi_Year_indicator/Sum:output:0*
T0*
_output_shapes
:?
Bsequential/dense_features/Multi_Year_indicator/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
Dsequential/dense_features/Multi_Year_indicator/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
Dsequential/dense_features/Multi_Year_indicator/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
<sequential/dense_features/Multi_Year_indicator/strided_sliceStridedSlice=sequential/dense_features/Multi_Year_indicator/Shape:output:0Ksequential/dense_features/Multi_Year_indicator/strided_slice/stack:output:0Msequential/dense_features/Multi_Year_indicator/strided_slice/stack_1:output:0Msequential/dense_features/Multi_Year_indicator/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
>sequential/dense_features/Multi_Year_indicator/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
<sequential/dense_features/Multi_Year_indicator/Reshape/shapePackEsequential/dense_features/Multi_Year_indicator/strided_slice:output:0Gsequential/dense_features/Multi_Year_indicator/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
6sequential/dense_features/Multi_Year_indicator/ReshapeReshape;sequential/dense_features/Multi_Year_indicator/Sum:output:0Esequential/dense_features/Multi_Year_indicator/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????h
,sequential/dense_features/N_Patent_App/ShapeShapen_patent_app*
T0*
_output_shapes
:?
:sequential/dense_features/N_Patent_App/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
<sequential/dense_features/N_Patent_App/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
<sequential/dense_features/N_Patent_App/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
4sequential/dense_features/N_Patent_App/strided_sliceStridedSlice5sequential/dense_features/N_Patent_App/Shape:output:0Csequential/dense_features/N_Patent_App/strided_slice/stack:output:0Esequential/dense_features/N_Patent_App/strided_slice/stack_1:output:0Esequential/dense_features/N_Patent_App/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskx
6sequential/dense_features/N_Patent_App/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
4sequential/dense_features/N_Patent_App/Reshape/shapePack=sequential/dense_features/N_Patent_App/strided_slice:output:0?sequential/dense_features/N_Patent_App/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
.sequential/dense_features/N_Patent_App/ReshapeReshapen_patent_app=sequential/dense_features/N_Patent_App/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????h
,sequential/dense_features/N_Patent_Reg/ShapeShapen_patent_reg*
T0*
_output_shapes
:?
:sequential/dense_features/N_Patent_Reg/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
<sequential/dense_features/N_Patent_Reg/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
<sequential/dense_features/N_Patent_Reg/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
4sequential/dense_features/N_Patent_Reg/strided_sliceStridedSlice5sequential/dense_features/N_Patent_Reg/Shape:output:0Csequential/dense_features/N_Patent_Reg/strided_slice/stack:output:0Esequential/dense_features/N_Patent_Reg/strided_slice/stack_1:output:0Esequential/dense_features/N_Patent_Reg/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskx
6sequential/dense_features/N_Patent_Reg/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
4sequential/dense_features/N_Patent_Reg/Reshape/shapePack=sequential/dense_features/N_Patent_Reg/strided_slice:output:0?sequential/dense_features/N_Patent_Reg/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
.sequential/dense_features/N_Patent_Reg/ReshapeReshapen_patent_reg=sequential/dense_features/N_Patent_Reg/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????t
2sequential/dense_features/N_of_Korean_Patent/ShapeShapen_of_korean_patent*
T0*
_output_shapes
:?
@sequential/dense_features/N_of_Korean_Patent/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
Bsequential/dense_features/N_of_Korean_Patent/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
Bsequential/dense_features/N_of_Korean_Patent/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
:sequential/dense_features/N_of_Korean_Patent/strided_sliceStridedSlice;sequential/dense_features/N_of_Korean_Patent/Shape:output:0Isequential/dense_features/N_of_Korean_Patent/strided_slice/stack:output:0Ksequential/dense_features/N_of_Korean_Patent/strided_slice/stack_1:output:0Ksequential/dense_features/N_of_Korean_Patent/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask~
<sequential/dense_features/N_of_Korean_Patent/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
:sequential/dense_features/N_of_Korean_Patent/Reshape/shapePackCsequential/dense_features/N_of_Korean_Patent/strided_slice:output:0Esequential/dense_features/N_of_Korean_Patent/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
4sequential/dense_features/N_of_Korean_Patent/ReshapeReshapen_of_korean_patentCsequential/dense_features/N_of_Korean_Patent/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????d
*sequential/dense_features/N_of_Paper/ShapeShape
n_of_paper*
T0*
_output_shapes
:?
8sequential/dense_features/N_of_Paper/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
:sequential/dense_features/N_of_Paper/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
:sequential/dense_features/N_of_Paper/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
2sequential/dense_features/N_of_Paper/strided_sliceStridedSlice3sequential/dense_features/N_of_Paper/Shape:output:0Asequential/dense_features/N_of_Paper/strided_slice/stack:output:0Csequential/dense_features/N_of_Paper/strided_slice/stack_1:output:0Csequential/dense_features/N_of_Paper/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskv
4sequential/dense_features/N_of_Paper/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
2sequential/dense_features/N_of_Paper/Reshape/shapePack;sequential/dense_features/N_of_Paper/strided_slice:output:0=sequential/dense_features/N_of_Paper/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
,sequential/dense_features/N_of_Paper/ReshapeReshape
n_of_paper;sequential/dense_features/N_of_Paper/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????`
(sequential/dense_features/N_of_SCI/ShapeShapen_of_sci*
T0*
_output_shapes
:?
6sequential/dense_features/N_of_SCI/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
8sequential/dense_features/N_of_SCI/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
8sequential/dense_features/N_of_SCI/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
0sequential/dense_features/N_of_SCI/strided_sliceStridedSlice1sequential/dense_features/N_of_SCI/Shape:output:0?sequential/dense_features/N_of_SCI/strided_slice/stack:output:0Asequential/dense_features/N_of_SCI/strided_slice/stack_1:output:0Asequential/dense_features/N_of_SCI/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskt
2sequential/dense_features/N_of_SCI/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
0sequential/dense_features/N_of_SCI/Reshape/shapePack9sequential/dense_features/N_of_SCI/strided_slice:output:0;sequential/dense_features/N_of_SCI/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
*sequential/dense_features/N_of_SCI/ReshapeReshapen_of_sci9sequential/dense_features/N_of_SCI/Reshape/shape:output:0*
T0*'
_output_shapes
:??????????
Vsequential/dense_features/National_Strategy_2_indicator/to_sparse_input/ignore_value/xConst*
_output_shapes
: *
dtype0*
valueB :
??????????
Tsequential/dense_features/National_Strategy_2_indicator/to_sparse_input/ignore_valueCast_sequential/dense_features/National_Strategy_2_indicator/to_sparse_input/ignore_value/x:output:0*

DstT0	*

SrcT0*
_output_shapes
: ?
Psequential/dense_features/National_Strategy_2_indicator/to_sparse_input/NotEqualNotEqualnational_strategy_2Xsequential/dense_features/National_Strategy_2_indicator/to_sparse_input/ignore_value:y:0*
T0	*'
_output_shapes
:??????????
Osequential/dense_features/National_Strategy_2_indicator/to_sparse_input/indicesWhereTsequential/dense_features/National_Strategy_2_indicator/to_sparse_input/NotEqual:z:0*'
_output_shapes
:??????????
Nsequential/dense_features/National_Strategy_2_indicator/to_sparse_input/valuesGatherNdnational_strategy_2Wsequential/dense_features/National_Strategy_2_indicator/to_sparse_input/indices:index:0*
Tindices0	*
Tparams0	*#
_output_shapes
:??????????
Ssequential/dense_features/National_Strategy_2_indicator/to_sparse_input/dense_shapeShapenational_strategy_2*
T0	*
_output_shapes
:*
out_type0	?
Usequential/dense_features/National_Strategy_2_indicator/None_Lookup/LookupTableFindV2LookupTableFindV2bsequential_dense_features_national_strategy_2_indicator_none_lookup_lookuptablefindv2_table_handleWsequential/dense_features/National_Strategy_2_indicator/to_sparse_input/values:output:0csequential_dense_features_national_strategy_2_indicator_none_lookup_lookuptablefindv2_default_value*	
Tin0	*

Tout0	*#
_output_shapes
:??????????
Ssequential/dense_features/National_Strategy_2_indicator/SparseToDense/default_valueConst*
_output_shapes
: *
dtype0	*
valueB	 R
??????????
Esequential/dense_features/National_Strategy_2_indicator/SparseToDenseSparseToDenseWsequential/dense_features/National_Strategy_2_indicator/to_sparse_input/indices:index:0\sequential/dense_features/National_Strategy_2_indicator/to_sparse_input/dense_shape:output:0^sequential/dense_features/National_Strategy_2_indicator/None_Lookup/LookupTableFindV2:values:0\sequential/dense_features/National_Strategy_2_indicator/SparseToDense/default_value:output:0*
T0	*
Tindices0	*'
_output_shapes
:??????????
Esequential/dense_features/National_Strategy_2_indicator/one_hot/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
Gsequential/dense_features/National_Strategy_2_indicator/one_hot/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    ?
Esequential/dense_features/National_Strategy_2_indicator/one_hot/depthConst*
_output_shapes
: *
dtype0*
value	B :?
?sequential/dense_features/National_Strategy_2_indicator/one_hotOneHotMsequential/dense_features/National_Strategy_2_indicator/SparseToDense:dense:0Nsequential/dense_features/National_Strategy_2_indicator/one_hot/depth:output:0Nsequential/dense_features/National_Strategy_2_indicator/one_hot/Const:output:0Psequential/dense_features/National_Strategy_2_indicator/one_hot/Const_1:output:0*
T0*+
_output_shapes
:??????????
Msequential/dense_features/National_Strategy_2_indicator/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
??????????
;sequential/dense_features/National_Strategy_2_indicator/SumSumHsequential/dense_features/National_Strategy_2_indicator/one_hot:output:0Vsequential/dense_features/National_Strategy_2_indicator/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:??????????
=sequential/dense_features/National_Strategy_2_indicator/ShapeShapeDsequential/dense_features/National_Strategy_2_indicator/Sum:output:0*
T0*
_output_shapes
:?
Ksequential/dense_features/National_Strategy_2_indicator/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
Msequential/dense_features/National_Strategy_2_indicator/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
Msequential/dense_features/National_Strategy_2_indicator/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
Esequential/dense_features/National_Strategy_2_indicator/strided_sliceStridedSliceFsequential/dense_features/National_Strategy_2_indicator/Shape:output:0Tsequential/dense_features/National_Strategy_2_indicator/strided_slice/stack:output:0Vsequential/dense_features/National_Strategy_2_indicator/strided_slice/stack_1:output:0Vsequential/dense_features/National_Strategy_2_indicator/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
Gsequential/dense_features/National_Strategy_2_indicator/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
Esequential/dense_features/National_Strategy_2_indicator/Reshape/shapePackNsequential/dense_features/National_Strategy_2_indicator/strided_slice:output:0Psequential/dense_features/National_Strategy_2_indicator/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
?sequential/dense_features/National_Strategy_2_indicator/ReshapeReshapeDsequential/dense_features/National_Strategy_2_indicator/Sum:output:0Nsequential/dense_features/National_Strategy_2_indicator/Reshape/shape:output:0*
T0*'
_output_shapes
:??????????
Jsequential/dense_features/RnD_Org_indicator/to_sparse_input/ignore_value/xConst*
_output_shapes
: *
dtype0*
valueB :
??????????
Hsequential/dense_features/RnD_Org_indicator/to_sparse_input/ignore_valueCastSsequential/dense_features/RnD_Org_indicator/to_sparse_input/ignore_value/x:output:0*

DstT0	*

SrcT0*
_output_shapes
: ?
Dsequential/dense_features/RnD_Org_indicator/to_sparse_input/NotEqualNotEqualrnd_orgLsequential/dense_features/RnD_Org_indicator/to_sparse_input/ignore_value:y:0*
T0	*'
_output_shapes
:??????????
Csequential/dense_features/RnD_Org_indicator/to_sparse_input/indicesWhereHsequential/dense_features/RnD_Org_indicator/to_sparse_input/NotEqual:z:0*'
_output_shapes
:??????????
Bsequential/dense_features/RnD_Org_indicator/to_sparse_input/valuesGatherNdrnd_orgKsequential/dense_features/RnD_Org_indicator/to_sparse_input/indices:index:0*
Tindices0	*
Tparams0	*#
_output_shapes
:??????????
Gsequential/dense_features/RnD_Org_indicator/to_sparse_input/dense_shapeShapernd_org*
T0	*
_output_shapes
:*
out_type0	?
Isequential/dense_features/RnD_Org_indicator/None_Lookup/LookupTableFindV2LookupTableFindV2Vsequential_dense_features_rnd_org_indicator_none_lookup_lookuptablefindv2_table_handleKsequential/dense_features/RnD_Org_indicator/to_sparse_input/values:output:0Wsequential_dense_features_rnd_org_indicator_none_lookup_lookuptablefindv2_default_value*	
Tin0	*

Tout0	*#
_output_shapes
:??????????
Gsequential/dense_features/RnD_Org_indicator/SparseToDense/default_valueConst*
_output_shapes
: *
dtype0	*
valueB	 R
??????????
9sequential/dense_features/RnD_Org_indicator/SparseToDenseSparseToDenseKsequential/dense_features/RnD_Org_indicator/to_sparse_input/indices:index:0Psequential/dense_features/RnD_Org_indicator/to_sparse_input/dense_shape:output:0Rsequential/dense_features/RnD_Org_indicator/None_Lookup/LookupTableFindV2:values:0Psequential/dense_features/RnD_Org_indicator/SparseToDense/default_value:output:0*
T0	*
Tindices0	*'
_output_shapes
:?????????~
9sequential/dense_features/RnD_Org_indicator/one_hot/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
;sequential/dense_features/RnD_Org_indicator/one_hot/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    {
9sequential/dense_features/RnD_Org_indicator/one_hot/depthConst*
_output_shapes
: *
dtype0*
value	B :?
3sequential/dense_features/RnD_Org_indicator/one_hotOneHotAsequential/dense_features/RnD_Org_indicator/SparseToDense:dense:0Bsequential/dense_features/RnD_Org_indicator/one_hot/depth:output:0Bsequential/dense_features/RnD_Org_indicator/one_hot/Const:output:0Dsequential/dense_features/RnD_Org_indicator/one_hot/Const_1:output:0*
T0*+
_output_shapes
:??????????
Asequential/dense_features/RnD_Org_indicator/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
??????????
/sequential/dense_features/RnD_Org_indicator/SumSum<sequential/dense_features/RnD_Org_indicator/one_hot:output:0Jsequential/dense_features/RnD_Org_indicator/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:??????????
1sequential/dense_features/RnD_Org_indicator/ShapeShape8sequential/dense_features/RnD_Org_indicator/Sum:output:0*
T0*
_output_shapes
:?
?sequential/dense_features/RnD_Org_indicator/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
Asequential/dense_features/RnD_Org_indicator/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
Asequential/dense_features/RnD_Org_indicator/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
9sequential/dense_features/RnD_Org_indicator/strided_sliceStridedSlice:sequential/dense_features/RnD_Org_indicator/Shape:output:0Hsequential/dense_features/RnD_Org_indicator/strided_slice/stack:output:0Jsequential/dense_features/RnD_Org_indicator/strided_slice/stack_1:output:0Jsequential/dense_features/RnD_Org_indicator/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask}
;sequential/dense_features/RnD_Org_indicator/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
9sequential/dense_features/RnD_Org_indicator/Reshape/shapePackBsequential/dense_features/RnD_Org_indicator/strided_slice:output:0Dsequential/dense_features/RnD_Org_indicator/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
3sequential/dense_features/RnD_Org_indicator/ReshapeReshape8sequential/dense_features/RnD_Org_indicator/Sum:output:0Bsequential/dense_features/RnD_Org_indicator/Reshape/shape:output:0*
T0*'
_output_shapes
:??????????
Lsequential/dense_features/RnD_Stage_indicator/to_sparse_input/ignore_value/xConst*
_output_shapes
: *
dtype0*
valueB :
??????????
Jsequential/dense_features/RnD_Stage_indicator/to_sparse_input/ignore_valueCastUsequential/dense_features/RnD_Stage_indicator/to_sparse_input/ignore_value/x:output:0*

DstT0	*

SrcT0*
_output_shapes
: ?
Fsequential/dense_features/RnD_Stage_indicator/to_sparse_input/NotEqualNotEqual	rnd_stageNsequential/dense_features/RnD_Stage_indicator/to_sparse_input/ignore_value:y:0*
T0	*'
_output_shapes
:??????????
Esequential/dense_features/RnD_Stage_indicator/to_sparse_input/indicesWhereJsequential/dense_features/RnD_Stage_indicator/to_sparse_input/NotEqual:z:0*'
_output_shapes
:??????????
Dsequential/dense_features/RnD_Stage_indicator/to_sparse_input/valuesGatherNd	rnd_stageMsequential/dense_features/RnD_Stage_indicator/to_sparse_input/indices:index:0*
Tindices0	*
Tparams0	*#
_output_shapes
:??????????
Isequential/dense_features/RnD_Stage_indicator/to_sparse_input/dense_shapeShape	rnd_stage*
T0	*
_output_shapes
:*
out_type0	?
Ksequential/dense_features/RnD_Stage_indicator/None_Lookup/LookupTableFindV2LookupTableFindV2Xsequential_dense_features_rnd_stage_indicator_none_lookup_lookuptablefindv2_table_handleMsequential/dense_features/RnD_Stage_indicator/to_sparse_input/values:output:0Ysequential_dense_features_rnd_stage_indicator_none_lookup_lookuptablefindv2_default_value*	
Tin0	*

Tout0	*#
_output_shapes
:??????????
Isequential/dense_features/RnD_Stage_indicator/SparseToDense/default_valueConst*
_output_shapes
: *
dtype0	*
valueB	 R
??????????
;sequential/dense_features/RnD_Stage_indicator/SparseToDenseSparseToDenseMsequential/dense_features/RnD_Stage_indicator/to_sparse_input/indices:index:0Rsequential/dense_features/RnD_Stage_indicator/to_sparse_input/dense_shape:output:0Tsequential/dense_features/RnD_Stage_indicator/None_Lookup/LookupTableFindV2:values:0Rsequential/dense_features/RnD_Stage_indicator/SparseToDense/default_value:output:0*
T0	*
Tindices0	*'
_output_shapes
:??????????
;sequential/dense_features/RnD_Stage_indicator/one_hot/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
=sequential/dense_features/RnD_Stage_indicator/one_hot/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    }
;sequential/dense_features/RnD_Stage_indicator/one_hot/depthConst*
_output_shapes
: *
dtype0*
value	B :?
5sequential/dense_features/RnD_Stage_indicator/one_hotOneHotCsequential/dense_features/RnD_Stage_indicator/SparseToDense:dense:0Dsequential/dense_features/RnD_Stage_indicator/one_hot/depth:output:0Dsequential/dense_features/RnD_Stage_indicator/one_hot/Const:output:0Fsequential/dense_features/RnD_Stage_indicator/one_hot/Const_1:output:0*
T0*+
_output_shapes
:??????????
Csequential/dense_features/RnD_Stage_indicator/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
??????????
1sequential/dense_features/RnD_Stage_indicator/SumSum>sequential/dense_features/RnD_Stage_indicator/one_hot:output:0Lsequential/dense_features/RnD_Stage_indicator/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:??????????
3sequential/dense_features/RnD_Stage_indicator/ShapeShape:sequential/dense_features/RnD_Stage_indicator/Sum:output:0*
T0*
_output_shapes
:?
Asequential/dense_features/RnD_Stage_indicator/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
Csequential/dense_features/RnD_Stage_indicator/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
Csequential/dense_features/RnD_Stage_indicator/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
;sequential/dense_features/RnD_Stage_indicator/strided_sliceStridedSlice<sequential/dense_features/RnD_Stage_indicator/Shape:output:0Jsequential/dense_features/RnD_Stage_indicator/strided_slice/stack:output:0Lsequential/dense_features/RnD_Stage_indicator/strided_slice/stack_1:output:0Lsequential/dense_features/RnD_Stage_indicator/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
=sequential/dense_features/RnD_Stage_indicator/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
;sequential/dense_features/RnD_Stage_indicator/Reshape/shapePackDsequential/dense_features/RnD_Stage_indicator/strided_slice:output:0Fsequential/dense_features/RnD_Stage_indicator/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
5sequential/dense_features/RnD_Stage_indicator/ReshapeReshape:sequential/dense_features/RnD_Stage_indicator/Sum:output:0Dsequential/dense_features/RnD_Stage_indicator/Reshape/shape:output:0*
T0*'
_output_shapes
:??????????
Nsequential/dense_features/STP_Code_11_indicator/to_sparse_input/ignore_value/xConst*
_output_shapes
: *
dtype0*
valueB B ?
Hsequential/dense_features/STP_Code_11_indicator/to_sparse_input/NotEqualNotEqualstp_code_11Wsequential/dense_features/STP_Code_11_indicator/to_sparse_input/ignore_value/x:output:0*
T0*'
_output_shapes
:??????????
Gsequential/dense_features/STP_Code_11_indicator/to_sparse_input/indicesWhereLsequential/dense_features/STP_Code_11_indicator/to_sparse_input/NotEqual:z:0*'
_output_shapes
:??????????
Fsequential/dense_features/STP_Code_11_indicator/to_sparse_input/valuesGatherNdstp_code_11Osequential/dense_features/STP_Code_11_indicator/to_sparse_input/indices:index:0*
Tindices0	*
Tparams0*#
_output_shapes
:??????????
Ksequential/dense_features/STP_Code_11_indicator/to_sparse_input/dense_shapeShapestp_code_11*
T0*
_output_shapes
:*
out_type0	?
Msequential/dense_features/STP_Code_11_indicator/None_Lookup/LookupTableFindV2LookupTableFindV2Zsequential_dense_features_stp_code_11_indicator_none_lookup_lookuptablefindv2_table_handleOsequential/dense_features/STP_Code_11_indicator/to_sparse_input/values:output:0[sequential_dense_features_stp_code_11_indicator_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:??????????
Ksequential/dense_features/STP_Code_11_indicator/SparseToDense/default_valueConst*
_output_shapes
: *
dtype0	*
valueB	 R
??????????
=sequential/dense_features/STP_Code_11_indicator/SparseToDenseSparseToDenseOsequential/dense_features/STP_Code_11_indicator/to_sparse_input/indices:index:0Tsequential/dense_features/STP_Code_11_indicator/to_sparse_input/dense_shape:output:0Vsequential/dense_features/STP_Code_11_indicator/None_Lookup/LookupTableFindV2:values:0Tsequential/dense_features/STP_Code_11_indicator/SparseToDense/default_value:output:0*
T0	*
Tindices0	*'
_output_shapes
:??????????
=sequential/dense_features/STP_Code_11_indicator/one_hot/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
?sequential/dense_features/STP_Code_11_indicator/one_hot/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    ?
=sequential/dense_features/STP_Code_11_indicator/one_hot/depthConst*
_output_shapes
: *
dtype0*
value
B :??
7sequential/dense_features/STP_Code_11_indicator/one_hotOneHotEsequential/dense_features/STP_Code_11_indicator/SparseToDense:dense:0Fsequential/dense_features/STP_Code_11_indicator/one_hot/depth:output:0Fsequential/dense_features/STP_Code_11_indicator/one_hot/Const:output:0Hsequential/dense_features/STP_Code_11_indicator/one_hot/Const_1:output:0*
T0*,
_output_shapes
:???????????
Esequential/dense_features/STP_Code_11_indicator/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
??????????
3sequential/dense_features/STP_Code_11_indicator/SumSum@sequential/dense_features/STP_Code_11_indicator/one_hot:output:0Nsequential/dense_features/STP_Code_11_indicator/Sum/reduction_indices:output:0*
T0*(
_output_shapes
:???????????
5sequential/dense_features/STP_Code_11_indicator/ShapeShape<sequential/dense_features/STP_Code_11_indicator/Sum:output:0*
T0*
_output_shapes
:?
Csequential/dense_features/STP_Code_11_indicator/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
Esequential/dense_features/STP_Code_11_indicator/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
Esequential/dense_features/STP_Code_11_indicator/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
=sequential/dense_features/STP_Code_11_indicator/strided_sliceStridedSlice>sequential/dense_features/STP_Code_11_indicator/Shape:output:0Lsequential/dense_features/STP_Code_11_indicator/strided_slice/stack:output:0Nsequential/dense_features/STP_Code_11_indicator/strided_slice/stack_1:output:0Nsequential/dense_features/STP_Code_11_indicator/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
?sequential/dense_features/STP_Code_11_indicator/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value
B :??
=sequential/dense_features/STP_Code_11_indicator/Reshape/shapePackFsequential/dense_features/STP_Code_11_indicator/strided_slice:output:0Hsequential/dense_features/STP_Code_11_indicator/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
7sequential/dense_features/STP_Code_11_indicator/ReshapeReshape<sequential/dense_features/STP_Code_11_indicator/Sum:output:0Fsequential/dense_features/STP_Code_11_indicator/Reshape/shape:output:0*
T0*(
_output_shapes
:??????????r
1sequential/dense_features/STP_Code_1_Weight/ShapeShapestp_code_1_weight*
T0*
_output_shapes
:?
?sequential/dense_features/STP_Code_1_Weight/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
Asequential/dense_features/STP_Code_1_Weight/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
Asequential/dense_features/STP_Code_1_Weight/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
9sequential/dense_features/STP_Code_1_Weight/strided_sliceStridedSlice:sequential/dense_features/STP_Code_1_Weight/Shape:output:0Hsequential/dense_features/STP_Code_1_Weight/strided_slice/stack:output:0Jsequential/dense_features/STP_Code_1_Weight/strided_slice/stack_1:output:0Jsequential/dense_features/STP_Code_1_Weight/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask}
;sequential/dense_features/STP_Code_1_Weight/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
9sequential/dense_features/STP_Code_1_Weight/Reshape/shapePackBsequential/dense_features/STP_Code_1_Weight/strided_slice:output:0Dsequential/dense_features/STP_Code_1_Weight/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
3sequential/dense_features/STP_Code_1_Weight/ReshapeReshapestp_code_1_weightBsequential/dense_features/STP_Code_1_Weight/Reshape/shape:output:0*
T0*'
_output_shapes
:??????????
Nsequential/dense_features/STP_Code_21_indicator/to_sparse_input/ignore_value/xConst*
_output_shapes
: *
dtype0*
valueB B ?
Hsequential/dense_features/STP_Code_21_indicator/to_sparse_input/NotEqualNotEqualstp_code_21Wsequential/dense_features/STP_Code_21_indicator/to_sparse_input/ignore_value/x:output:0*
T0*'
_output_shapes
:??????????
Gsequential/dense_features/STP_Code_21_indicator/to_sparse_input/indicesWhereLsequential/dense_features/STP_Code_21_indicator/to_sparse_input/NotEqual:z:0*'
_output_shapes
:??????????
Fsequential/dense_features/STP_Code_21_indicator/to_sparse_input/valuesGatherNdstp_code_21Osequential/dense_features/STP_Code_21_indicator/to_sparse_input/indices:index:0*
Tindices0	*
Tparams0*#
_output_shapes
:??????????
Ksequential/dense_features/STP_Code_21_indicator/to_sparse_input/dense_shapeShapestp_code_21*
T0*
_output_shapes
:*
out_type0	?
Msequential/dense_features/STP_Code_21_indicator/None_Lookup/LookupTableFindV2LookupTableFindV2Zsequential_dense_features_stp_code_21_indicator_none_lookup_lookuptablefindv2_table_handleOsequential/dense_features/STP_Code_21_indicator/to_sparse_input/values:output:0[sequential_dense_features_stp_code_21_indicator_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:??????????
Ksequential/dense_features/STP_Code_21_indicator/SparseToDense/default_valueConst*
_output_shapes
: *
dtype0	*
valueB	 R
??????????
=sequential/dense_features/STP_Code_21_indicator/SparseToDenseSparseToDenseOsequential/dense_features/STP_Code_21_indicator/to_sparse_input/indices:index:0Tsequential/dense_features/STP_Code_21_indicator/to_sparse_input/dense_shape:output:0Vsequential/dense_features/STP_Code_21_indicator/None_Lookup/LookupTableFindV2:values:0Tsequential/dense_features/STP_Code_21_indicator/SparseToDense/default_value:output:0*
T0	*
Tindices0	*'
_output_shapes
:??????????
=sequential/dense_features/STP_Code_21_indicator/one_hot/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
?sequential/dense_features/STP_Code_21_indicator/one_hot/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    ?
=sequential/dense_features/STP_Code_21_indicator/one_hot/depthConst*
_output_shapes
: *
dtype0*
value
B :??
7sequential/dense_features/STP_Code_21_indicator/one_hotOneHotEsequential/dense_features/STP_Code_21_indicator/SparseToDense:dense:0Fsequential/dense_features/STP_Code_21_indicator/one_hot/depth:output:0Fsequential/dense_features/STP_Code_21_indicator/one_hot/Const:output:0Hsequential/dense_features/STP_Code_21_indicator/one_hot/Const_1:output:0*
T0*,
_output_shapes
:???????????
Esequential/dense_features/STP_Code_21_indicator/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
??????????
3sequential/dense_features/STP_Code_21_indicator/SumSum@sequential/dense_features/STP_Code_21_indicator/one_hot:output:0Nsequential/dense_features/STP_Code_21_indicator/Sum/reduction_indices:output:0*
T0*(
_output_shapes
:???????????
5sequential/dense_features/STP_Code_21_indicator/ShapeShape<sequential/dense_features/STP_Code_21_indicator/Sum:output:0*
T0*
_output_shapes
:?
Csequential/dense_features/STP_Code_21_indicator/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
Esequential/dense_features/STP_Code_21_indicator/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
Esequential/dense_features/STP_Code_21_indicator/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
=sequential/dense_features/STP_Code_21_indicator/strided_sliceStridedSlice>sequential/dense_features/STP_Code_21_indicator/Shape:output:0Lsequential/dense_features/STP_Code_21_indicator/strided_slice/stack:output:0Nsequential/dense_features/STP_Code_21_indicator/strided_slice/stack_1:output:0Nsequential/dense_features/STP_Code_21_indicator/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
?sequential/dense_features/STP_Code_21_indicator/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value
B :??
=sequential/dense_features/STP_Code_21_indicator/Reshape/shapePackFsequential/dense_features/STP_Code_21_indicator/strided_slice:output:0Hsequential/dense_features/STP_Code_21_indicator/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
7sequential/dense_features/STP_Code_21_indicator/ReshapeReshape<sequential/dense_features/STP_Code_21_indicator/Sum:output:0Fsequential/dense_features/STP_Code_21_indicator/Reshape/shape:output:0*
T0*(
_output_shapes
:??????????r
1sequential/dense_features/STP_Code_2_Weight/ShapeShapestp_code_2_weight*
T0*
_output_shapes
:?
?sequential/dense_features/STP_Code_2_Weight/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
Asequential/dense_features/STP_Code_2_Weight/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
Asequential/dense_features/STP_Code_2_Weight/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
9sequential/dense_features/STP_Code_2_Weight/strided_sliceStridedSlice:sequential/dense_features/STP_Code_2_Weight/Shape:output:0Hsequential/dense_features/STP_Code_2_Weight/strided_slice/stack:output:0Jsequential/dense_features/STP_Code_2_Weight/strided_slice/stack_1:output:0Jsequential/dense_features/STP_Code_2_Weight/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask}
;sequential/dense_features/STP_Code_2_Weight/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
9sequential/dense_features/STP_Code_2_Weight/Reshape/shapePackBsequential/dense_features/STP_Code_2_Weight/strided_slice:output:0Dsequential/dense_features/STP_Code_2_Weight/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
3sequential/dense_features/STP_Code_2_Weight/ReshapeReshapestp_code_2_weightBsequential/dense_features/STP_Code_2_Weight/Reshape/shape:output:0*
T0*'
_output_shapes
:??????????
Isequential/dense_features/SixT_2_indicator/to_sparse_input/ignore_value/xConst*
_output_shapes
: *
dtype0*
valueB :
??????????
Gsequential/dense_features/SixT_2_indicator/to_sparse_input/ignore_valueCastRsequential/dense_features/SixT_2_indicator/to_sparse_input/ignore_value/x:output:0*

DstT0	*

SrcT0*
_output_shapes
: ?
Csequential/dense_features/SixT_2_indicator/to_sparse_input/NotEqualNotEqualsixt_2Ksequential/dense_features/SixT_2_indicator/to_sparse_input/ignore_value:y:0*
T0	*'
_output_shapes
:??????????
Bsequential/dense_features/SixT_2_indicator/to_sparse_input/indicesWhereGsequential/dense_features/SixT_2_indicator/to_sparse_input/NotEqual:z:0*'
_output_shapes
:??????????
Asequential/dense_features/SixT_2_indicator/to_sparse_input/valuesGatherNdsixt_2Jsequential/dense_features/SixT_2_indicator/to_sparse_input/indices:index:0*
Tindices0	*
Tparams0	*#
_output_shapes
:??????????
Fsequential/dense_features/SixT_2_indicator/to_sparse_input/dense_shapeShapesixt_2*
T0	*
_output_shapes
:*
out_type0	?
Hsequential/dense_features/SixT_2_indicator/None_Lookup/LookupTableFindV2LookupTableFindV2Usequential_dense_features_sixt_2_indicator_none_lookup_lookuptablefindv2_table_handleJsequential/dense_features/SixT_2_indicator/to_sparse_input/values:output:0Vsequential_dense_features_sixt_2_indicator_none_lookup_lookuptablefindv2_default_value*	
Tin0	*

Tout0	*#
_output_shapes
:??????????
Fsequential/dense_features/SixT_2_indicator/SparseToDense/default_valueConst*
_output_shapes
: *
dtype0	*
valueB	 R
??????????
8sequential/dense_features/SixT_2_indicator/SparseToDenseSparseToDenseJsequential/dense_features/SixT_2_indicator/to_sparse_input/indices:index:0Osequential/dense_features/SixT_2_indicator/to_sparse_input/dense_shape:output:0Qsequential/dense_features/SixT_2_indicator/None_Lookup/LookupTableFindV2:values:0Osequential/dense_features/SixT_2_indicator/SparseToDense/default_value:output:0*
T0	*
Tindices0	*'
_output_shapes
:?????????}
8sequential/dense_features/SixT_2_indicator/one_hot/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??
:sequential/dense_features/SixT_2_indicator/one_hot/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    z
8sequential/dense_features/SixT_2_indicator/one_hot/depthConst*
_output_shapes
: *
dtype0*
value	B :?
2sequential/dense_features/SixT_2_indicator/one_hotOneHot@sequential/dense_features/SixT_2_indicator/SparseToDense:dense:0Asequential/dense_features/SixT_2_indicator/one_hot/depth:output:0Asequential/dense_features/SixT_2_indicator/one_hot/Const:output:0Csequential/dense_features/SixT_2_indicator/one_hot/Const_1:output:0*
T0*+
_output_shapes
:??????????
@sequential/dense_features/SixT_2_indicator/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
??????????
.sequential/dense_features/SixT_2_indicator/SumSum;sequential/dense_features/SixT_2_indicator/one_hot:output:0Isequential/dense_features/SixT_2_indicator/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:??????????
0sequential/dense_features/SixT_2_indicator/ShapeShape7sequential/dense_features/SixT_2_indicator/Sum:output:0*
T0*
_output_shapes
:?
>sequential/dense_features/SixT_2_indicator/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
@sequential/dense_features/SixT_2_indicator/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
@sequential/dense_features/SixT_2_indicator/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
8sequential/dense_features/SixT_2_indicator/strided_sliceStridedSlice9sequential/dense_features/SixT_2_indicator/Shape:output:0Gsequential/dense_features/SixT_2_indicator/strided_slice/stack:output:0Isequential/dense_features/SixT_2_indicator/strided_slice/stack_1:output:0Isequential/dense_features/SixT_2_indicator/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask|
:sequential/dense_features/SixT_2_indicator/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
8sequential/dense_features/SixT_2_indicator/Reshape/shapePackAsequential/dense_features/SixT_2_indicator/strided_slice:output:0Csequential/dense_features/SixT_2_indicator/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
2sequential/dense_features/SixT_2_indicator/ReshapeReshape7sequential/dense_features/SixT_2_indicator/Sum:output:0Asequential/dense_features/SixT_2_indicator/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????X
$sequential/dense_features/Year/ShapeShapeyear*
T0*
_output_shapes
:|
2sequential/dense_features/Year/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ~
4sequential/dense_features/Year/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:~
4sequential/dense_features/Year/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
,sequential/dense_features/Year/strided_sliceStridedSlice-sequential/dense_features/Year/Shape:output:0;sequential/dense_features/Year/strided_slice/stack:output:0=sequential/dense_features/Year/strided_slice/stack_1:output:0=sequential/dense_features/Year/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskp
.sequential/dense_features/Year/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
,sequential/dense_features/Year/Reshape/shapePack5sequential/dense_features/Year/strided_slice:output:07sequential/dense_features/Year/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
&sequential/dense_features/Year/ReshapeReshapeyear5sequential/dense_features/Year/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????p
%sequential/dense_features/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
??????????
 sequential/dense_features/concatConcatV2Dsequential/dense_features/Application_Area_1_Weight/Reshape:output:0Gsequential/dense_features/Application_Area_1_indicator/Reshape:output:0Dsequential/dense_features/Application_Area_2_Weight/Reshape:output:0Gsequential/dense_features/Application_Area_2_indicator/Reshape:output:0Bsequential/dense_features/Cowork_Abroad_indicator/Reshape:output:0?sequential/dense_features/Cowork_Cor_indicator/Reshape:output:0@sequential/dense_features/Cowork_Inst_indicator/Reshape:output:0?sequential/dense_features/Cowork_Uni_indicator/Reshape:output:0?sequential/dense_features/Cowork_etc_indicator/Reshape:output:0@sequential/dense_features/Econ_Social_indicator/Reshape:output:0?sequential/dense_features/Green_Tech_indicator/Reshape:output:07sequential/dense_features/Log_Duration/Reshape:output:07sequential/dense_features/Log_RnD_Fund/Reshape:output:0?sequential/dense_features/Multi_Year_indicator/Reshape:output:07sequential/dense_features/N_Patent_App/Reshape:output:07sequential/dense_features/N_Patent_Reg/Reshape:output:0=sequential/dense_features/N_of_Korean_Patent/Reshape:output:05sequential/dense_features/N_of_Paper/Reshape:output:03sequential/dense_features/N_of_SCI/Reshape:output:0Hsequential/dense_features/National_Strategy_2_indicator/Reshape:output:0<sequential/dense_features/RnD_Org_indicator/Reshape:output:0>sequential/dense_features/RnD_Stage_indicator/Reshape:output:0@sequential/dense_features/STP_Code_11_indicator/Reshape:output:0<sequential/dense_features/STP_Code_1_Weight/Reshape:output:0@sequential/dense_features/STP_Code_21_indicator/Reshape:output:0<sequential/dense_features/STP_Code_2_Weight/Reshape:output:0;sequential/dense_features/SixT_2_indicator/Reshape:output:0/sequential/dense_features/Year/Reshape:output:0.sequential/dense_features/concat/axis:output:0*
N*
T0*(
_output_shapes
:???????????
&sequential/dense/MatMul/ReadVariableOpReadVariableOp/sequential_dense_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype0?
sequential/dense/MatMulMatMul)sequential/dense_features/concat:output:0.sequential/dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
'sequential/dense/BiasAdd/ReadVariableOpReadVariableOp0sequential_dense_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
sequential/dense/BiasAddBiasAdd!sequential/dense/MatMul:product:0/sequential/dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????s
sequential/dense/ReluRelu!sequential/dense/BiasAdd:output:0*
T0*(
_output_shapes
:??????????
sequential/dropout/IdentityIdentity#sequential/dense/Relu:activations:0*
T0*(
_output_shapes
:???????????
(sequential/dense_1/MatMul/ReadVariableOpReadVariableOp1sequential_dense_1_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype0?
sequential/dense_1/MatMulMatMul$sequential/dropout/Identity:output:00sequential/dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
)sequential/dense_1/BiasAdd/ReadVariableOpReadVariableOp2sequential_dense_1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
sequential/dense_1/BiasAddBiasAdd#sequential/dense_1/MatMul:product:01sequential/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????w
sequential/dense_1/ReluRelu#sequential/dense_1/BiasAdd:output:0*
T0*(
_output_shapes
:???????????
sequential/dropout_1/IdentityIdentity%sequential/dense_1/Relu:activations:0*
T0*(
_output_shapes
:???????????
(sequential/dense_2/MatMul/ReadVariableOpReadVariableOp1sequential_dense_2_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype0?
sequential/dense_2/MatMulMatMul&sequential/dropout_1/Identity:output:00sequential/dense_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
)sequential/dense_2/BiasAdd/ReadVariableOpReadVariableOp2sequential_dense_2_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
sequential/dense_2/BiasAddBiasAdd#sequential/dense_2/MatMul:product:01sequential/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????w
sequential/dense_2/ReluRelu#sequential/dense_2/BiasAdd:output:0*
T0*(
_output_shapes
:???????????
sequential/dropout_2/IdentityIdentity%sequential/dense_2/Relu:activations:0*
T0*(
_output_shapes
:???????????
(sequential/dense_3/MatMul/ReadVariableOpReadVariableOp1sequential_dense_3_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype0?
sequential/dense_3/MatMulMatMul&sequential/dropout_2/Identity:output:00sequential/dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
)sequential/dense_3/BiasAdd/ReadVariableOpReadVariableOp2sequential_dense_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
sequential/dense_3/BiasAddBiasAdd#sequential/dense_3/MatMul:product:01sequential/dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????|
sequential/dense_3/SigmoidSigmoid#sequential/dense_3/BiasAdd:output:0*
T0*'
_output_shapes
:?????????m
IdentityIdentitysequential/dense_3/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp(^sequential/dense/BiasAdd/ReadVariableOp'^sequential/dense/MatMul/ReadVariableOp*^sequential/dense_1/BiasAdd/ReadVariableOp)^sequential/dense_1/MatMul/ReadVariableOp*^sequential/dense_2/BiasAdd/ReadVariableOp)^sequential/dense_2/MatMul/ReadVariableOp*^sequential/dense_3/BiasAdd/ReadVariableOp)^sequential/dense_3/MatMul/ReadVariableOpU^sequential/dense_features/Application_Area_1_indicator/None_Lookup/LookupTableFindV2U^sequential/dense_features/Application_Area_2_indicator/None_Lookup/LookupTableFindV2P^sequential/dense_features/Cowork_Abroad_indicator/None_Lookup/LookupTableFindV2M^sequential/dense_features/Cowork_Cor_indicator/None_Lookup/LookupTableFindV2N^sequential/dense_features/Cowork_Inst_indicator/None_Lookup/LookupTableFindV2M^sequential/dense_features/Cowork_Uni_indicator/None_Lookup/LookupTableFindV2M^sequential/dense_features/Cowork_etc_indicator/None_Lookup/LookupTableFindV2N^sequential/dense_features/Econ_Social_indicator/None_Lookup/LookupTableFindV2M^sequential/dense_features/Green_Tech_indicator/None_Lookup/LookupTableFindV2M^sequential/dense_features/Multi_Year_indicator/None_Lookup/LookupTableFindV2V^sequential/dense_features/National_Strategy_2_indicator/None_Lookup/LookupTableFindV2J^sequential/dense_features/RnD_Org_indicator/None_Lookup/LookupTableFindV2L^sequential/dense_features/RnD_Stage_indicator/None_Lookup/LookupTableFindV2N^sequential/dense_features/STP_Code_11_indicator/None_Lookup/LookupTableFindV2N^sequential/dense_features/STP_Code_21_indicator/None_Lookup/LookupTableFindV2I^sequential/dense_features/SixT_2_indicator/None_Lookup/LookupTableFindV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2R
'sequential/dense/BiasAdd/ReadVariableOp'sequential/dense/BiasAdd/ReadVariableOp2P
&sequential/dense/MatMul/ReadVariableOp&sequential/dense/MatMul/ReadVariableOp2V
)sequential/dense_1/BiasAdd/ReadVariableOp)sequential/dense_1/BiasAdd/ReadVariableOp2T
(sequential/dense_1/MatMul/ReadVariableOp(sequential/dense_1/MatMul/ReadVariableOp2V
)sequential/dense_2/BiasAdd/ReadVariableOp)sequential/dense_2/BiasAdd/ReadVariableOp2T
(sequential/dense_2/MatMul/ReadVariableOp(sequential/dense_2/MatMul/ReadVariableOp2V
)sequential/dense_3/BiasAdd/ReadVariableOp)sequential/dense_3/BiasAdd/ReadVariableOp2T
(sequential/dense_3/MatMul/ReadVariableOp(sequential/dense_3/MatMul/ReadVariableOp2?
Tsequential/dense_features/Application_Area_1_indicator/None_Lookup/LookupTableFindV2Tsequential/dense_features/Application_Area_1_indicator/None_Lookup/LookupTableFindV22?
Tsequential/dense_features/Application_Area_2_indicator/None_Lookup/LookupTableFindV2Tsequential/dense_features/Application_Area_2_indicator/None_Lookup/LookupTableFindV22?
Osequential/dense_features/Cowork_Abroad_indicator/None_Lookup/LookupTableFindV2Osequential/dense_features/Cowork_Abroad_indicator/None_Lookup/LookupTableFindV22?
Lsequential/dense_features/Cowork_Cor_indicator/None_Lookup/LookupTableFindV2Lsequential/dense_features/Cowork_Cor_indicator/None_Lookup/LookupTableFindV22?
Msequential/dense_features/Cowork_Inst_indicator/None_Lookup/LookupTableFindV2Msequential/dense_features/Cowork_Inst_indicator/None_Lookup/LookupTableFindV22?
Lsequential/dense_features/Cowork_Uni_indicator/None_Lookup/LookupTableFindV2Lsequential/dense_features/Cowork_Uni_indicator/None_Lookup/LookupTableFindV22?
Lsequential/dense_features/Cowork_etc_indicator/None_Lookup/LookupTableFindV2Lsequential/dense_features/Cowork_etc_indicator/None_Lookup/LookupTableFindV22?
Msequential/dense_features/Econ_Social_indicator/None_Lookup/LookupTableFindV2Msequential/dense_features/Econ_Social_indicator/None_Lookup/LookupTableFindV22?
Lsequential/dense_features/Green_Tech_indicator/None_Lookup/LookupTableFindV2Lsequential/dense_features/Green_Tech_indicator/None_Lookup/LookupTableFindV22?
Lsequential/dense_features/Multi_Year_indicator/None_Lookup/LookupTableFindV2Lsequential/dense_features/Multi_Year_indicator/None_Lookup/LookupTableFindV22?
Usequential/dense_features/National_Strategy_2_indicator/None_Lookup/LookupTableFindV2Usequential/dense_features/National_Strategy_2_indicator/None_Lookup/LookupTableFindV22?
Isequential/dense_features/RnD_Org_indicator/None_Lookup/LookupTableFindV2Isequential/dense_features/RnD_Org_indicator/None_Lookup/LookupTableFindV22?
Ksequential/dense_features/RnD_Stage_indicator/None_Lookup/LookupTableFindV2Ksequential/dense_features/RnD_Stage_indicator/None_Lookup/LookupTableFindV22?
Msequential/dense_features/STP_Code_11_indicator/None_Lookup/LookupTableFindV2Msequential/dense_features/STP_Code_11_indicator/None_Lookup/LookupTableFindV22?
Msequential/dense_features/STP_Code_21_indicator/None_Lookup/LookupTableFindV2Msequential/dense_features/STP_Code_21_indicator/None_Lookup/LookupTableFindV22?
Hsequential/dense_features/SixT_2_indicator/None_Lookup/LookupTableFindV2Hsequential/dense_features/SixT_2_indicator/None_Lookup/LookupTableFindV2:[ W
'
_output_shapes
:?????????
,
_user_specified_nameApplication_Area_1:b^
'
_output_shapes
:?????????
3
_user_specified_nameApplication_Area_1_Weight:[W
'
_output_shapes
:?????????
,
_user_specified_nameApplication_Area_2:b^
'
_output_shapes
:?????????
3
_user_specified_nameApplication_Area_2_Weight:VR
'
_output_shapes
:?????????
'
_user_specified_nameCowork_Abroad:SO
'
_output_shapes
:?????????
$
_user_specified_name
Cowork_Cor:TP
'
_output_shapes
:?????????
%
_user_specified_nameCowork_Inst:SO
'
_output_shapes
:?????????
$
_user_specified_name
Cowork_Uni:SO
'
_output_shapes
:?????????
$
_user_specified_name
Cowork_etc:T	P
'
_output_shapes
:?????????
%
_user_specified_nameEcon_Social:S
O
'
_output_shapes
:?????????
$
_user_specified_name
Green_Tech:UQ
'
_output_shapes
:?????????
&
_user_specified_nameLog_Duration:UQ
'
_output_shapes
:?????????
&
_user_specified_nameLog_RnD_Fund:SO
'
_output_shapes
:?????????
$
_user_specified_name
Multi_Year:UQ
'
_output_shapes
:?????????
&
_user_specified_nameN_Patent_App:UQ
'
_output_shapes
:?????????
&
_user_specified_nameN_Patent_Reg:[W
'
_output_shapes
:?????????
,
_user_specified_nameN_of_Korean_Patent:SO
'
_output_shapes
:?????????
$
_user_specified_name
N_of_Paper:QM
'
_output_shapes
:?????????
"
_user_specified_name
N_of_SCI:\X
'
_output_shapes
:?????????
-
_user_specified_nameNational_Strategy_2:PL
'
_output_shapes
:?????????
!
_user_specified_name	RnD_Org:RN
'
_output_shapes
:?????????
#
_user_specified_name	RnD_Stage:TP
'
_output_shapes
:?????????
%
_user_specified_nameSTP_Code_11:ZV
'
_output_shapes
:?????????
+
_user_specified_nameSTP_Code_1_Weight:TP
'
_output_shapes
:?????????
%
_user_specified_nameSTP_Code_21:ZV
'
_output_shapes
:?????????
+
_user_specified_nameSTP_Code_2_Weight:OK
'
_output_shapes
:?????????
 
_user_specified_nameSixT_2:MI
'
_output_shapes
:?????????

_user_specified_nameYear:

_output_shapes
: :

_output_shapes
: :!

_output_shapes
: :#

_output_shapes
: :%

_output_shapes
: :'

_output_shapes
: :)

_output_shapes
: :+

_output_shapes
: :-

_output_shapes
: :/

_output_shapes
: :1

_output_shapes
: :3

_output_shapes
: :5

_output_shapes
: :7

_output_shapes
: :9

_output_shapes
: :;

_output_shapes
: 
?
?
__inference__initializer_1337892
.table_init547_lookuptableimportv2_table_handle*
&table_init547_lookuptableimportv2_keys	,
(table_init547_lookuptableimportv2_values	
identity??!table_init547/LookupTableImportV2?
!table_init547/LookupTableImportV2LookupTableImportV2.table_init547_lookuptableimportv2_table_handle&table_init547_lookuptableimportv2_keys(table_init547_lookuptableimportv2_values*	
Tin0	*

Tout0	*
_output_shapes
 G
ConstConst*
_output_shapes
: *
dtype0*
value	B :L
IdentityIdentityConst:output:0^NoOp*
T0*
_output_shapes
: j
NoOpNoOp"^table_init547/LookupTableImportV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*!
_input_shapes
: ::2F
!table_init547/LookupTableImportV2!table_init547/LookupTableImportV2: 

_output_shapes
:: 

_output_shapes
:
?
-
__inference__destroyer_133776
identityG
ConstConst*
_output_shapes
: *
dtype0*
value	B :E
IdentityIdentityConst:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 
?
-
__inference__destroyer_133812
identityG
ConstConst*
_output_shapes
: *
dtype0*
value	B :E
IdentityIdentityConst:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 
?
c
*__inference_dropout_2_layer_call_fn_133613

inputs
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_dropout_2_layer_call_and_return_conditional_losses_129305p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:??????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:??????????22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
__inference__initializer_1338432
.table_init711_lookuptableimportv2_table_handle*
&table_init711_lookuptableimportv2_keys	,
(table_init711_lookuptableimportv2_values	
identity??!table_init711/LookupTableImportV2?
!table_init711/LookupTableImportV2LookupTableImportV2.table_init711_lookuptableimportv2_table_handle&table_init711_lookuptableimportv2_keys(table_init711_lookuptableimportv2_values*	
Tin0	*

Tout0	*
_output_shapes
 G
ConstConst*
_output_shapes
: *
dtype0*
value	B :L
IdentityIdentityConst:output:0^NoOp*
T0*
_output_shapes
: j
NoOpNoOp"^table_init711/LookupTableImportV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*!
_input_shapes
: ::2F
!table_init711/LookupTableImportV2!table_init711/LookupTableImportV2: 

_output_shapes
:: 

_output_shapes
:
??
?
J__inference_dense_features_layer_call_and_return_conditional_losses_132969
features_application_area_1&
"features_application_area_1_weight
features_application_area_2&
"features_application_area_2_weight
features_cowork_abroad
features_cowork_cor
features_cowork_inst
features_cowork_uni
features_cowork_etc
features_econ_social	
features_green_tech	
features_log_duration
features_log_rnd_fund
features_multi_year	
features_n_patent_app
features_n_patent_reg
features_n_of_korean_patent
features_n_of_paper
features_n_of_sci 
features_national_strategy_2	
features_rnd_org	
features_rnd_stage	
features_stp_code_11
features_stp_code_1_weight
features_stp_code_21
features_stp_code_2_weight
features_sixt_2	
features_yearK
Gapplication_area_1_indicator_none_lookup_lookuptablefindv2_table_handleL
Happlication_area_1_indicator_none_lookup_lookuptablefindv2_default_value	K
Gapplication_area_2_indicator_none_lookup_lookuptablefindv2_table_handleL
Happlication_area_2_indicator_none_lookup_lookuptablefindv2_default_value	F
Bcowork_abroad_indicator_none_lookup_lookuptablefindv2_table_handleG
Ccowork_abroad_indicator_none_lookup_lookuptablefindv2_default_value	C
?cowork_cor_indicator_none_lookup_lookuptablefindv2_table_handleD
@cowork_cor_indicator_none_lookup_lookuptablefindv2_default_value	D
@cowork_inst_indicator_none_lookup_lookuptablefindv2_table_handleE
Acowork_inst_indicator_none_lookup_lookuptablefindv2_default_value	C
?cowork_uni_indicator_none_lookup_lookuptablefindv2_table_handleD
@cowork_uni_indicator_none_lookup_lookuptablefindv2_default_value	C
?cowork_etc_indicator_none_lookup_lookuptablefindv2_table_handleD
@cowork_etc_indicator_none_lookup_lookuptablefindv2_default_value	D
@econ_social_indicator_none_lookup_lookuptablefindv2_table_handleE
Aecon_social_indicator_none_lookup_lookuptablefindv2_default_value	C
?green_tech_indicator_none_lookup_lookuptablefindv2_table_handleD
@green_tech_indicator_none_lookup_lookuptablefindv2_default_value	C
?multi_year_indicator_none_lookup_lookuptablefindv2_table_handleD
@multi_year_indicator_none_lookup_lookuptablefindv2_default_value	L
Hnational_strategy_2_indicator_none_lookup_lookuptablefindv2_table_handleM
Inational_strategy_2_indicator_none_lookup_lookuptablefindv2_default_value	@
<rnd_org_indicator_none_lookup_lookuptablefindv2_table_handleA
=rnd_org_indicator_none_lookup_lookuptablefindv2_default_value	B
>rnd_stage_indicator_none_lookup_lookuptablefindv2_table_handleC
?rnd_stage_indicator_none_lookup_lookuptablefindv2_default_value	D
@stp_code_11_indicator_none_lookup_lookuptablefindv2_table_handleE
Astp_code_11_indicator_none_lookup_lookuptablefindv2_default_value	D
@stp_code_21_indicator_none_lookup_lookuptablefindv2_table_handleE
Astp_code_21_indicator_none_lookup_lookuptablefindv2_default_value	?
;sixt_2_indicator_none_lookup_lookuptablefindv2_table_handle@
<sixt_2_indicator_none_lookup_lookuptablefindv2_default_value	
identity??:Application_Area_1_indicator/None_Lookup/LookupTableFindV2?:Application_Area_2_indicator/None_Lookup/LookupTableFindV2?5Cowork_Abroad_indicator/None_Lookup/LookupTableFindV2?2Cowork_Cor_indicator/None_Lookup/LookupTableFindV2?3Cowork_Inst_indicator/None_Lookup/LookupTableFindV2?2Cowork_Uni_indicator/None_Lookup/LookupTableFindV2?2Cowork_etc_indicator/None_Lookup/LookupTableFindV2?3Econ_Social_indicator/None_Lookup/LookupTableFindV2?2Green_Tech_indicator/None_Lookup/LookupTableFindV2?2Multi_Year_indicator/None_Lookup/LookupTableFindV2?;National_Strategy_2_indicator/None_Lookup/LookupTableFindV2?/RnD_Org_indicator/None_Lookup/LookupTableFindV2?1RnD_Stage_indicator/None_Lookup/LookupTableFindV2?3STP_Code_11_indicator/None_Lookup/LookupTableFindV2?3STP_Code_21_indicator/None_Lookup/LookupTableFindV2?.SixT_2_indicator/None_Lookup/LookupTableFindV2q
Application_Area_1_Weight/ShapeShape"features_application_area_1_weight*
T0*
_output_shapes
:w
-Application_Area_1_Weight/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: y
/Application_Area_1_Weight/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:y
/Application_Area_1_Weight/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
'Application_Area_1_Weight/strided_sliceStridedSlice(Application_Area_1_Weight/Shape:output:06Application_Area_1_Weight/strided_slice/stack:output:08Application_Area_1_Weight/strided_slice/stack_1:output:08Application_Area_1_Weight/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskk
)Application_Area_1_Weight/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
'Application_Area_1_Weight/Reshape/shapePack0Application_Area_1_Weight/strided_slice:output:02Application_Area_1_Weight/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
!Application_Area_1_Weight/ReshapeReshape"features_application_area_1_weight0Application_Area_1_Weight/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????|
;Application_Area_1_indicator/to_sparse_input/ignore_value/xConst*
_output_shapes
: *
dtype0*
valueB B ?
5Application_Area_1_indicator/to_sparse_input/NotEqualNotEqualfeatures_application_area_1DApplication_Area_1_indicator/to_sparse_input/ignore_value/x:output:0*
T0*'
_output_shapes
:??????????
4Application_Area_1_indicator/to_sparse_input/indicesWhere9Application_Area_1_indicator/to_sparse_input/NotEqual:z:0*'
_output_shapes
:??????????
3Application_Area_1_indicator/to_sparse_input/valuesGatherNdfeatures_application_area_1<Application_Area_1_indicator/to_sparse_input/indices:index:0*
Tindices0	*
Tparams0*#
_output_shapes
:??????????
8Application_Area_1_indicator/to_sparse_input/dense_shapeShapefeatures_application_area_1*
T0*
_output_shapes
:*
out_type0	?
:Application_Area_1_indicator/None_Lookup/LookupTableFindV2LookupTableFindV2Gapplication_area_1_indicator_none_lookup_lookuptablefindv2_table_handle<Application_Area_1_indicator/to_sparse_input/values:output:0Happlication_area_1_indicator_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:??????????
8Application_Area_1_indicator/SparseToDense/default_valueConst*
_output_shapes
: *
dtype0	*
valueB	 R
??????????
*Application_Area_1_indicator/SparseToDenseSparseToDense<Application_Area_1_indicator/to_sparse_input/indices:index:0AApplication_Area_1_indicator/to_sparse_input/dense_shape:output:0CApplication_Area_1_indicator/None_Lookup/LookupTableFindV2:values:0AApplication_Area_1_indicator/SparseToDense/default_value:output:0*
T0	*
Tindices0	*'
_output_shapes
:?????????o
*Application_Area_1_indicator/one_hot/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??q
,Application_Area_1_indicator/one_hot/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    l
*Application_Area_1_indicator/one_hot/depthConst*
_output_shapes
: *
dtype0*
value	B :!?
$Application_Area_1_indicator/one_hotOneHot2Application_Area_1_indicator/SparseToDense:dense:03Application_Area_1_indicator/one_hot/depth:output:03Application_Area_1_indicator/one_hot/Const:output:05Application_Area_1_indicator/one_hot/Const_1:output:0*
T0*+
_output_shapes
:?????????!?
2Application_Area_1_indicator/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
??????????
 Application_Area_1_indicator/SumSum-Application_Area_1_indicator/one_hot:output:0;Application_Area_1_indicator/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:?????????!{
"Application_Area_1_indicator/ShapeShape)Application_Area_1_indicator/Sum:output:0*
T0*
_output_shapes
:z
0Application_Area_1_indicator/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: |
2Application_Area_1_indicator/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:|
2Application_Area_1_indicator/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
*Application_Area_1_indicator/strided_sliceStridedSlice+Application_Area_1_indicator/Shape:output:09Application_Area_1_indicator/strided_slice/stack:output:0;Application_Area_1_indicator/strided_slice/stack_1:output:0;Application_Area_1_indicator/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskn
,Application_Area_1_indicator/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :!?
*Application_Area_1_indicator/Reshape/shapePack3Application_Area_1_indicator/strided_slice:output:05Application_Area_1_indicator/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
$Application_Area_1_indicator/ReshapeReshape)Application_Area_1_indicator/Sum:output:03Application_Area_1_indicator/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????!q
Application_Area_2_Weight/ShapeShape"features_application_area_2_weight*
T0*
_output_shapes
:w
-Application_Area_2_Weight/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: y
/Application_Area_2_Weight/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:y
/Application_Area_2_Weight/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
'Application_Area_2_Weight/strided_sliceStridedSlice(Application_Area_2_Weight/Shape:output:06Application_Area_2_Weight/strided_slice/stack:output:08Application_Area_2_Weight/strided_slice/stack_1:output:08Application_Area_2_Weight/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskk
)Application_Area_2_Weight/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
'Application_Area_2_Weight/Reshape/shapePack0Application_Area_2_Weight/strided_slice:output:02Application_Area_2_Weight/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
!Application_Area_2_Weight/ReshapeReshape"features_application_area_2_weight0Application_Area_2_Weight/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????|
;Application_Area_2_indicator/to_sparse_input/ignore_value/xConst*
_output_shapes
: *
dtype0*
valueB B ?
5Application_Area_2_indicator/to_sparse_input/NotEqualNotEqualfeatures_application_area_2DApplication_Area_2_indicator/to_sparse_input/ignore_value/x:output:0*
T0*'
_output_shapes
:??????????
4Application_Area_2_indicator/to_sparse_input/indicesWhere9Application_Area_2_indicator/to_sparse_input/NotEqual:z:0*'
_output_shapes
:??????????
3Application_Area_2_indicator/to_sparse_input/valuesGatherNdfeatures_application_area_2<Application_Area_2_indicator/to_sparse_input/indices:index:0*
Tindices0	*
Tparams0*#
_output_shapes
:??????????
8Application_Area_2_indicator/to_sparse_input/dense_shapeShapefeatures_application_area_2*
T0*
_output_shapes
:*
out_type0	?
:Application_Area_2_indicator/None_Lookup/LookupTableFindV2LookupTableFindV2Gapplication_area_2_indicator_none_lookup_lookuptablefindv2_table_handle<Application_Area_2_indicator/to_sparse_input/values:output:0Happlication_area_2_indicator_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:??????????
8Application_Area_2_indicator/SparseToDense/default_valueConst*
_output_shapes
: *
dtype0	*
valueB	 R
??????????
*Application_Area_2_indicator/SparseToDenseSparseToDense<Application_Area_2_indicator/to_sparse_input/indices:index:0AApplication_Area_2_indicator/to_sparse_input/dense_shape:output:0CApplication_Area_2_indicator/None_Lookup/LookupTableFindV2:values:0AApplication_Area_2_indicator/SparseToDense/default_value:output:0*
T0	*
Tindices0	*'
_output_shapes
:?????????o
*Application_Area_2_indicator/one_hot/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??q
,Application_Area_2_indicator/one_hot/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    l
*Application_Area_2_indicator/one_hot/depthConst*
_output_shapes
: *
dtype0*
value	B :"?
$Application_Area_2_indicator/one_hotOneHot2Application_Area_2_indicator/SparseToDense:dense:03Application_Area_2_indicator/one_hot/depth:output:03Application_Area_2_indicator/one_hot/Const:output:05Application_Area_2_indicator/one_hot/Const_1:output:0*
T0*+
_output_shapes
:?????????"?
2Application_Area_2_indicator/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
??????????
 Application_Area_2_indicator/SumSum-Application_Area_2_indicator/one_hot:output:0;Application_Area_2_indicator/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:?????????"{
"Application_Area_2_indicator/ShapeShape)Application_Area_2_indicator/Sum:output:0*
T0*
_output_shapes
:z
0Application_Area_2_indicator/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: |
2Application_Area_2_indicator/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:|
2Application_Area_2_indicator/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
*Application_Area_2_indicator/strided_sliceStridedSlice+Application_Area_2_indicator/Shape:output:09Application_Area_2_indicator/strided_slice/stack:output:0;Application_Area_2_indicator/strided_slice/stack_1:output:0;Application_Area_2_indicator/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskn
,Application_Area_2_indicator/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :"?
*Application_Area_2_indicator/Reshape/shapePack3Application_Area_2_indicator/strided_slice:output:05Application_Area_2_indicator/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
$Application_Area_2_indicator/ReshapeReshape)Application_Area_2_indicator/Sum:output:03Application_Area_2_indicator/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????"w
6Cowork_Abroad_indicator/to_sparse_input/ignore_value/xConst*
_output_shapes
: *
dtype0*
valueB B ?
0Cowork_Abroad_indicator/to_sparse_input/NotEqualNotEqualfeatures_cowork_abroad?Cowork_Abroad_indicator/to_sparse_input/ignore_value/x:output:0*
T0*'
_output_shapes
:??????????
/Cowork_Abroad_indicator/to_sparse_input/indicesWhere4Cowork_Abroad_indicator/to_sparse_input/NotEqual:z:0*'
_output_shapes
:??????????
.Cowork_Abroad_indicator/to_sparse_input/valuesGatherNdfeatures_cowork_abroad7Cowork_Abroad_indicator/to_sparse_input/indices:index:0*
Tindices0	*
Tparams0*#
_output_shapes
:??????????
3Cowork_Abroad_indicator/to_sparse_input/dense_shapeShapefeatures_cowork_abroad*
T0*
_output_shapes
:*
out_type0	?
5Cowork_Abroad_indicator/None_Lookup/LookupTableFindV2LookupTableFindV2Bcowork_abroad_indicator_none_lookup_lookuptablefindv2_table_handle7Cowork_Abroad_indicator/to_sparse_input/values:output:0Ccowork_abroad_indicator_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:?????????~
3Cowork_Abroad_indicator/SparseToDense/default_valueConst*
_output_shapes
: *
dtype0	*
valueB	 R
??????????
%Cowork_Abroad_indicator/SparseToDenseSparseToDense7Cowork_Abroad_indicator/to_sparse_input/indices:index:0<Cowork_Abroad_indicator/to_sparse_input/dense_shape:output:0>Cowork_Abroad_indicator/None_Lookup/LookupTableFindV2:values:0<Cowork_Abroad_indicator/SparseToDense/default_value:output:0*
T0	*
Tindices0	*'
_output_shapes
:?????????j
%Cowork_Abroad_indicator/one_hot/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??l
'Cowork_Abroad_indicator/one_hot/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    g
%Cowork_Abroad_indicator/one_hot/depthConst*
_output_shapes
: *
dtype0*
value	B :?
Cowork_Abroad_indicator/one_hotOneHot-Cowork_Abroad_indicator/SparseToDense:dense:0.Cowork_Abroad_indicator/one_hot/depth:output:0.Cowork_Abroad_indicator/one_hot/Const:output:00Cowork_Abroad_indicator/one_hot/Const_1:output:0*
T0*+
_output_shapes
:??????????
-Cowork_Abroad_indicator/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
??????????
Cowork_Abroad_indicator/SumSum(Cowork_Abroad_indicator/one_hot:output:06Cowork_Abroad_indicator/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:?????????q
Cowork_Abroad_indicator/ShapeShape$Cowork_Abroad_indicator/Sum:output:0*
T0*
_output_shapes
:u
+Cowork_Abroad_indicator/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: w
-Cowork_Abroad_indicator/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:w
-Cowork_Abroad_indicator/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
%Cowork_Abroad_indicator/strided_sliceStridedSlice&Cowork_Abroad_indicator/Shape:output:04Cowork_Abroad_indicator/strided_slice/stack:output:06Cowork_Abroad_indicator/strided_slice/stack_1:output:06Cowork_Abroad_indicator/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maski
'Cowork_Abroad_indicator/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
%Cowork_Abroad_indicator/Reshape/shapePack.Cowork_Abroad_indicator/strided_slice:output:00Cowork_Abroad_indicator/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
Cowork_Abroad_indicator/ReshapeReshape$Cowork_Abroad_indicator/Sum:output:0.Cowork_Abroad_indicator/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????t
3Cowork_Cor_indicator/to_sparse_input/ignore_value/xConst*
_output_shapes
: *
dtype0*
valueB B ?
-Cowork_Cor_indicator/to_sparse_input/NotEqualNotEqualfeatures_cowork_cor<Cowork_Cor_indicator/to_sparse_input/ignore_value/x:output:0*
T0*'
_output_shapes
:??????????
,Cowork_Cor_indicator/to_sparse_input/indicesWhere1Cowork_Cor_indicator/to_sparse_input/NotEqual:z:0*'
_output_shapes
:??????????
+Cowork_Cor_indicator/to_sparse_input/valuesGatherNdfeatures_cowork_cor4Cowork_Cor_indicator/to_sparse_input/indices:index:0*
Tindices0	*
Tparams0*#
_output_shapes
:??????????
0Cowork_Cor_indicator/to_sparse_input/dense_shapeShapefeatures_cowork_cor*
T0*
_output_shapes
:*
out_type0	?
2Cowork_Cor_indicator/None_Lookup/LookupTableFindV2LookupTableFindV2?cowork_cor_indicator_none_lookup_lookuptablefindv2_table_handle4Cowork_Cor_indicator/to_sparse_input/values:output:0@cowork_cor_indicator_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:?????????{
0Cowork_Cor_indicator/SparseToDense/default_valueConst*
_output_shapes
: *
dtype0	*
valueB	 R
??????????
"Cowork_Cor_indicator/SparseToDenseSparseToDense4Cowork_Cor_indicator/to_sparse_input/indices:index:09Cowork_Cor_indicator/to_sparse_input/dense_shape:output:0;Cowork_Cor_indicator/None_Lookup/LookupTableFindV2:values:09Cowork_Cor_indicator/SparseToDense/default_value:output:0*
T0	*
Tindices0	*'
_output_shapes
:?????????g
"Cowork_Cor_indicator/one_hot/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??i
$Cowork_Cor_indicator/one_hot/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    d
"Cowork_Cor_indicator/one_hot/depthConst*
_output_shapes
: *
dtype0*
value	B :?
Cowork_Cor_indicator/one_hotOneHot*Cowork_Cor_indicator/SparseToDense:dense:0+Cowork_Cor_indicator/one_hot/depth:output:0+Cowork_Cor_indicator/one_hot/Const:output:0-Cowork_Cor_indicator/one_hot/Const_1:output:0*
T0*+
_output_shapes
:?????????}
*Cowork_Cor_indicator/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
??????????
Cowork_Cor_indicator/SumSum%Cowork_Cor_indicator/one_hot:output:03Cowork_Cor_indicator/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:?????????k
Cowork_Cor_indicator/ShapeShape!Cowork_Cor_indicator/Sum:output:0*
T0*
_output_shapes
:r
(Cowork_Cor_indicator/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: t
*Cowork_Cor_indicator/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:t
*Cowork_Cor_indicator/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
"Cowork_Cor_indicator/strided_sliceStridedSlice#Cowork_Cor_indicator/Shape:output:01Cowork_Cor_indicator/strided_slice/stack:output:03Cowork_Cor_indicator/strided_slice/stack_1:output:03Cowork_Cor_indicator/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
$Cowork_Cor_indicator/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
"Cowork_Cor_indicator/Reshape/shapePack+Cowork_Cor_indicator/strided_slice:output:0-Cowork_Cor_indicator/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
Cowork_Cor_indicator/ReshapeReshape!Cowork_Cor_indicator/Sum:output:0+Cowork_Cor_indicator/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????u
4Cowork_Inst_indicator/to_sparse_input/ignore_value/xConst*
_output_shapes
: *
dtype0*
valueB B ?
.Cowork_Inst_indicator/to_sparse_input/NotEqualNotEqualfeatures_cowork_inst=Cowork_Inst_indicator/to_sparse_input/ignore_value/x:output:0*
T0*'
_output_shapes
:??????????
-Cowork_Inst_indicator/to_sparse_input/indicesWhere2Cowork_Inst_indicator/to_sparse_input/NotEqual:z:0*'
_output_shapes
:??????????
,Cowork_Inst_indicator/to_sparse_input/valuesGatherNdfeatures_cowork_inst5Cowork_Inst_indicator/to_sparse_input/indices:index:0*
Tindices0	*
Tparams0*#
_output_shapes
:??????????
1Cowork_Inst_indicator/to_sparse_input/dense_shapeShapefeatures_cowork_inst*
T0*
_output_shapes
:*
out_type0	?
3Cowork_Inst_indicator/None_Lookup/LookupTableFindV2LookupTableFindV2@cowork_inst_indicator_none_lookup_lookuptablefindv2_table_handle5Cowork_Inst_indicator/to_sparse_input/values:output:0Acowork_inst_indicator_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:?????????|
1Cowork_Inst_indicator/SparseToDense/default_valueConst*
_output_shapes
: *
dtype0	*
valueB	 R
??????????
#Cowork_Inst_indicator/SparseToDenseSparseToDense5Cowork_Inst_indicator/to_sparse_input/indices:index:0:Cowork_Inst_indicator/to_sparse_input/dense_shape:output:0<Cowork_Inst_indicator/None_Lookup/LookupTableFindV2:values:0:Cowork_Inst_indicator/SparseToDense/default_value:output:0*
T0	*
Tindices0	*'
_output_shapes
:?????????h
#Cowork_Inst_indicator/one_hot/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??j
%Cowork_Inst_indicator/one_hot/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    e
#Cowork_Inst_indicator/one_hot/depthConst*
_output_shapes
: *
dtype0*
value	B :?
Cowork_Inst_indicator/one_hotOneHot+Cowork_Inst_indicator/SparseToDense:dense:0,Cowork_Inst_indicator/one_hot/depth:output:0,Cowork_Inst_indicator/one_hot/Const:output:0.Cowork_Inst_indicator/one_hot/Const_1:output:0*
T0*+
_output_shapes
:?????????~
+Cowork_Inst_indicator/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
??????????
Cowork_Inst_indicator/SumSum&Cowork_Inst_indicator/one_hot:output:04Cowork_Inst_indicator/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:?????????m
Cowork_Inst_indicator/ShapeShape"Cowork_Inst_indicator/Sum:output:0*
T0*
_output_shapes
:s
)Cowork_Inst_indicator/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: u
+Cowork_Inst_indicator/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+Cowork_Inst_indicator/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
#Cowork_Inst_indicator/strided_sliceStridedSlice$Cowork_Inst_indicator/Shape:output:02Cowork_Inst_indicator/strided_slice/stack:output:04Cowork_Inst_indicator/strided_slice/stack_1:output:04Cowork_Inst_indicator/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskg
%Cowork_Inst_indicator/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
#Cowork_Inst_indicator/Reshape/shapePack,Cowork_Inst_indicator/strided_slice:output:0.Cowork_Inst_indicator/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
Cowork_Inst_indicator/ReshapeReshape"Cowork_Inst_indicator/Sum:output:0,Cowork_Inst_indicator/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????t
3Cowork_Uni_indicator/to_sparse_input/ignore_value/xConst*
_output_shapes
: *
dtype0*
valueB B ?
-Cowork_Uni_indicator/to_sparse_input/NotEqualNotEqualfeatures_cowork_uni<Cowork_Uni_indicator/to_sparse_input/ignore_value/x:output:0*
T0*'
_output_shapes
:??????????
,Cowork_Uni_indicator/to_sparse_input/indicesWhere1Cowork_Uni_indicator/to_sparse_input/NotEqual:z:0*'
_output_shapes
:??????????
+Cowork_Uni_indicator/to_sparse_input/valuesGatherNdfeatures_cowork_uni4Cowork_Uni_indicator/to_sparse_input/indices:index:0*
Tindices0	*
Tparams0*#
_output_shapes
:??????????
0Cowork_Uni_indicator/to_sparse_input/dense_shapeShapefeatures_cowork_uni*
T0*
_output_shapes
:*
out_type0	?
2Cowork_Uni_indicator/None_Lookup/LookupTableFindV2LookupTableFindV2?cowork_uni_indicator_none_lookup_lookuptablefindv2_table_handle4Cowork_Uni_indicator/to_sparse_input/values:output:0@cowork_uni_indicator_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:?????????{
0Cowork_Uni_indicator/SparseToDense/default_valueConst*
_output_shapes
: *
dtype0	*
valueB	 R
??????????
"Cowork_Uni_indicator/SparseToDenseSparseToDense4Cowork_Uni_indicator/to_sparse_input/indices:index:09Cowork_Uni_indicator/to_sparse_input/dense_shape:output:0;Cowork_Uni_indicator/None_Lookup/LookupTableFindV2:values:09Cowork_Uni_indicator/SparseToDense/default_value:output:0*
T0	*
Tindices0	*'
_output_shapes
:?????????g
"Cowork_Uni_indicator/one_hot/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??i
$Cowork_Uni_indicator/one_hot/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    d
"Cowork_Uni_indicator/one_hot/depthConst*
_output_shapes
: *
dtype0*
value	B :?
Cowork_Uni_indicator/one_hotOneHot*Cowork_Uni_indicator/SparseToDense:dense:0+Cowork_Uni_indicator/one_hot/depth:output:0+Cowork_Uni_indicator/one_hot/Const:output:0-Cowork_Uni_indicator/one_hot/Const_1:output:0*
T0*+
_output_shapes
:?????????}
*Cowork_Uni_indicator/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
??????????
Cowork_Uni_indicator/SumSum%Cowork_Uni_indicator/one_hot:output:03Cowork_Uni_indicator/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:?????????k
Cowork_Uni_indicator/ShapeShape!Cowork_Uni_indicator/Sum:output:0*
T0*
_output_shapes
:r
(Cowork_Uni_indicator/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: t
*Cowork_Uni_indicator/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:t
*Cowork_Uni_indicator/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
"Cowork_Uni_indicator/strided_sliceStridedSlice#Cowork_Uni_indicator/Shape:output:01Cowork_Uni_indicator/strided_slice/stack:output:03Cowork_Uni_indicator/strided_slice/stack_1:output:03Cowork_Uni_indicator/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
$Cowork_Uni_indicator/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
"Cowork_Uni_indicator/Reshape/shapePack+Cowork_Uni_indicator/strided_slice:output:0-Cowork_Uni_indicator/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
Cowork_Uni_indicator/ReshapeReshape!Cowork_Uni_indicator/Sum:output:0+Cowork_Uni_indicator/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????t
3Cowork_etc_indicator/to_sparse_input/ignore_value/xConst*
_output_shapes
: *
dtype0*
valueB B ?
-Cowork_etc_indicator/to_sparse_input/NotEqualNotEqualfeatures_cowork_etc<Cowork_etc_indicator/to_sparse_input/ignore_value/x:output:0*
T0*'
_output_shapes
:??????????
,Cowork_etc_indicator/to_sparse_input/indicesWhere1Cowork_etc_indicator/to_sparse_input/NotEqual:z:0*'
_output_shapes
:??????????
+Cowork_etc_indicator/to_sparse_input/valuesGatherNdfeatures_cowork_etc4Cowork_etc_indicator/to_sparse_input/indices:index:0*
Tindices0	*
Tparams0*#
_output_shapes
:??????????
0Cowork_etc_indicator/to_sparse_input/dense_shapeShapefeatures_cowork_etc*
T0*
_output_shapes
:*
out_type0	?
2Cowork_etc_indicator/None_Lookup/LookupTableFindV2LookupTableFindV2?cowork_etc_indicator_none_lookup_lookuptablefindv2_table_handle4Cowork_etc_indicator/to_sparse_input/values:output:0@cowork_etc_indicator_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:?????????{
0Cowork_etc_indicator/SparseToDense/default_valueConst*
_output_shapes
: *
dtype0	*
valueB	 R
??????????
"Cowork_etc_indicator/SparseToDenseSparseToDense4Cowork_etc_indicator/to_sparse_input/indices:index:09Cowork_etc_indicator/to_sparse_input/dense_shape:output:0;Cowork_etc_indicator/None_Lookup/LookupTableFindV2:values:09Cowork_etc_indicator/SparseToDense/default_value:output:0*
T0	*
Tindices0	*'
_output_shapes
:?????????g
"Cowork_etc_indicator/one_hot/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??i
$Cowork_etc_indicator/one_hot/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    d
"Cowork_etc_indicator/one_hot/depthConst*
_output_shapes
: *
dtype0*
value	B :?
Cowork_etc_indicator/one_hotOneHot*Cowork_etc_indicator/SparseToDense:dense:0+Cowork_etc_indicator/one_hot/depth:output:0+Cowork_etc_indicator/one_hot/Const:output:0-Cowork_etc_indicator/one_hot/Const_1:output:0*
T0*+
_output_shapes
:?????????}
*Cowork_etc_indicator/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
??????????
Cowork_etc_indicator/SumSum%Cowork_etc_indicator/one_hot:output:03Cowork_etc_indicator/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:?????????k
Cowork_etc_indicator/ShapeShape!Cowork_etc_indicator/Sum:output:0*
T0*
_output_shapes
:r
(Cowork_etc_indicator/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: t
*Cowork_etc_indicator/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:t
*Cowork_etc_indicator/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
"Cowork_etc_indicator/strided_sliceStridedSlice#Cowork_etc_indicator/Shape:output:01Cowork_etc_indicator/strided_slice/stack:output:03Cowork_etc_indicator/strided_slice/stack_1:output:03Cowork_etc_indicator/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
$Cowork_etc_indicator/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
"Cowork_etc_indicator/Reshape/shapePack+Cowork_etc_indicator/strided_slice:output:0-Cowork_etc_indicator/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
Cowork_etc_indicator/ReshapeReshape!Cowork_etc_indicator/Sum:output:0+Cowork_etc_indicator/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????
4Econ_Social_indicator/to_sparse_input/ignore_value/xConst*
_output_shapes
: *
dtype0*
valueB :
??????????
2Econ_Social_indicator/to_sparse_input/ignore_valueCast=Econ_Social_indicator/to_sparse_input/ignore_value/x:output:0*

DstT0	*

SrcT0*
_output_shapes
: ?
.Econ_Social_indicator/to_sparse_input/NotEqualNotEqualfeatures_econ_social6Econ_Social_indicator/to_sparse_input/ignore_value:y:0*
T0	*'
_output_shapes
:??????????
-Econ_Social_indicator/to_sparse_input/indicesWhere2Econ_Social_indicator/to_sparse_input/NotEqual:z:0*'
_output_shapes
:??????????
,Econ_Social_indicator/to_sparse_input/valuesGatherNdfeatures_econ_social5Econ_Social_indicator/to_sparse_input/indices:index:0*
Tindices0	*
Tparams0	*#
_output_shapes
:??????????
1Econ_Social_indicator/to_sparse_input/dense_shapeShapefeatures_econ_social*
T0	*
_output_shapes
:*
out_type0	?
3Econ_Social_indicator/None_Lookup/LookupTableFindV2LookupTableFindV2@econ_social_indicator_none_lookup_lookuptablefindv2_table_handle5Econ_Social_indicator/to_sparse_input/values:output:0Aecon_social_indicator_none_lookup_lookuptablefindv2_default_value*	
Tin0	*

Tout0	*#
_output_shapes
:?????????|
1Econ_Social_indicator/SparseToDense/default_valueConst*
_output_shapes
: *
dtype0	*
valueB	 R
??????????
#Econ_Social_indicator/SparseToDenseSparseToDense5Econ_Social_indicator/to_sparse_input/indices:index:0:Econ_Social_indicator/to_sparse_input/dense_shape:output:0<Econ_Social_indicator/None_Lookup/LookupTableFindV2:values:0:Econ_Social_indicator/SparseToDense/default_value:output:0*
T0	*
Tindices0	*'
_output_shapes
:?????????h
#Econ_Social_indicator/one_hot/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??j
%Econ_Social_indicator/one_hot/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    e
#Econ_Social_indicator/one_hot/depthConst*
_output_shapes
: *
dtype0*
value	B :?
Econ_Social_indicator/one_hotOneHot+Econ_Social_indicator/SparseToDense:dense:0,Econ_Social_indicator/one_hot/depth:output:0,Econ_Social_indicator/one_hot/Const:output:0.Econ_Social_indicator/one_hot/Const_1:output:0*
T0*+
_output_shapes
:?????????~
+Econ_Social_indicator/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
??????????
Econ_Social_indicator/SumSum&Econ_Social_indicator/one_hot:output:04Econ_Social_indicator/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:?????????m
Econ_Social_indicator/ShapeShape"Econ_Social_indicator/Sum:output:0*
T0*
_output_shapes
:s
)Econ_Social_indicator/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: u
+Econ_Social_indicator/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+Econ_Social_indicator/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
#Econ_Social_indicator/strided_sliceStridedSlice$Econ_Social_indicator/Shape:output:02Econ_Social_indicator/strided_slice/stack:output:04Econ_Social_indicator/strided_slice/stack_1:output:04Econ_Social_indicator/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskg
%Econ_Social_indicator/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
#Econ_Social_indicator/Reshape/shapePack,Econ_Social_indicator/strided_slice:output:0.Econ_Social_indicator/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
Econ_Social_indicator/ReshapeReshape"Econ_Social_indicator/Sum:output:0,Econ_Social_indicator/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????~
3Green_Tech_indicator/to_sparse_input/ignore_value/xConst*
_output_shapes
: *
dtype0*
valueB :
??????????
1Green_Tech_indicator/to_sparse_input/ignore_valueCast<Green_Tech_indicator/to_sparse_input/ignore_value/x:output:0*

DstT0	*

SrcT0*
_output_shapes
: ?
-Green_Tech_indicator/to_sparse_input/NotEqualNotEqualfeatures_green_tech5Green_Tech_indicator/to_sparse_input/ignore_value:y:0*
T0	*'
_output_shapes
:??????????
,Green_Tech_indicator/to_sparse_input/indicesWhere1Green_Tech_indicator/to_sparse_input/NotEqual:z:0*'
_output_shapes
:??????????
+Green_Tech_indicator/to_sparse_input/valuesGatherNdfeatures_green_tech4Green_Tech_indicator/to_sparse_input/indices:index:0*
Tindices0	*
Tparams0	*#
_output_shapes
:??????????
0Green_Tech_indicator/to_sparse_input/dense_shapeShapefeatures_green_tech*
T0	*
_output_shapes
:*
out_type0	?
2Green_Tech_indicator/None_Lookup/LookupTableFindV2LookupTableFindV2?green_tech_indicator_none_lookup_lookuptablefindv2_table_handle4Green_Tech_indicator/to_sparse_input/values:output:0@green_tech_indicator_none_lookup_lookuptablefindv2_default_value*	
Tin0	*

Tout0	*#
_output_shapes
:?????????{
0Green_Tech_indicator/SparseToDense/default_valueConst*
_output_shapes
: *
dtype0	*
valueB	 R
??????????
"Green_Tech_indicator/SparseToDenseSparseToDense4Green_Tech_indicator/to_sparse_input/indices:index:09Green_Tech_indicator/to_sparse_input/dense_shape:output:0;Green_Tech_indicator/None_Lookup/LookupTableFindV2:values:09Green_Tech_indicator/SparseToDense/default_value:output:0*
T0	*
Tindices0	*'
_output_shapes
:?????????g
"Green_Tech_indicator/one_hot/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??i
$Green_Tech_indicator/one_hot/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    d
"Green_Tech_indicator/one_hot/depthConst*
_output_shapes
: *
dtype0*
value	B :<?
Green_Tech_indicator/one_hotOneHot*Green_Tech_indicator/SparseToDense:dense:0+Green_Tech_indicator/one_hot/depth:output:0+Green_Tech_indicator/one_hot/Const:output:0-Green_Tech_indicator/one_hot/Const_1:output:0*
T0*+
_output_shapes
:?????????<}
*Green_Tech_indicator/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
??????????
Green_Tech_indicator/SumSum%Green_Tech_indicator/one_hot:output:03Green_Tech_indicator/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:?????????<k
Green_Tech_indicator/ShapeShape!Green_Tech_indicator/Sum:output:0*
T0*
_output_shapes
:r
(Green_Tech_indicator/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: t
*Green_Tech_indicator/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:t
*Green_Tech_indicator/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
"Green_Tech_indicator/strided_sliceStridedSlice#Green_Tech_indicator/Shape:output:01Green_Tech_indicator/strided_slice/stack:output:03Green_Tech_indicator/strided_slice/stack_1:output:03Green_Tech_indicator/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
$Green_Tech_indicator/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :<?
"Green_Tech_indicator/Reshape/shapePack+Green_Tech_indicator/strided_slice:output:0-Green_Tech_indicator/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
Green_Tech_indicator/ReshapeReshape!Green_Tech_indicator/Sum:output:0+Green_Tech_indicator/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????<W
Log_Duration/ShapeShapefeatures_log_duration*
T0*
_output_shapes
:j
 Log_Duration/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: l
"Log_Duration/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:l
"Log_Duration/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
Log_Duration/strided_sliceStridedSliceLog_Duration/Shape:output:0)Log_Duration/strided_slice/stack:output:0+Log_Duration/strided_slice/stack_1:output:0+Log_Duration/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask^
Log_Duration/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
Log_Duration/Reshape/shapePack#Log_Duration/strided_slice:output:0%Log_Duration/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
Log_Duration/ReshapeReshapefeatures_log_duration#Log_Duration/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????W
Log_RnD_Fund/ShapeShapefeatures_log_rnd_fund*
T0*
_output_shapes
:j
 Log_RnD_Fund/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: l
"Log_RnD_Fund/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:l
"Log_RnD_Fund/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
Log_RnD_Fund/strided_sliceStridedSliceLog_RnD_Fund/Shape:output:0)Log_RnD_Fund/strided_slice/stack:output:0+Log_RnD_Fund/strided_slice/stack_1:output:0+Log_RnD_Fund/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask^
Log_RnD_Fund/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
Log_RnD_Fund/Reshape/shapePack#Log_RnD_Fund/strided_slice:output:0%Log_RnD_Fund/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
Log_RnD_Fund/ReshapeReshapefeatures_log_rnd_fund#Log_RnD_Fund/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????~
3Multi_Year_indicator/to_sparse_input/ignore_value/xConst*
_output_shapes
: *
dtype0*
valueB :
??????????
1Multi_Year_indicator/to_sparse_input/ignore_valueCast<Multi_Year_indicator/to_sparse_input/ignore_value/x:output:0*

DstT0	*

SrcT0*
_output_shapes
: ?
-Multi_Year_indicator/to_sparse_input/NotEqualNotEqualfeatures_multi_year5Multi_Year_indicator/to_sparse_input/ignore_value:y:0*
T0	*'
_output_shapes
:??????????
,Multi_Year_indicator/to_sparse_input/indicesWhere1Multi_Year_indicator/to_sparse_input/NotEqual:z:0*'
_output_shapes
:??????????
+Multi_Year_indicator/to_sparse_input/valuesGatherNdfeatures_multi_year4Multi_Year_indicator/to_sparse_input/indices:index:0*
Tindices0	*
Tparams0	*#
_output_shapes
:??????????
0Multi_Year_indicator/to_sparse_input/dense_shapeShapefeatures_multi_year*
T0	*
_output_shapes
:*
out_type0	?
2Multi_Year_indicator/None_Lookup/LookupTableFindV2LookupTableFindV2?multi_year_indicator_none_lookup_lookuptablefindv2_table_handle4Multi_Year_indicator/to_sparse_input/values:output:0@multi_year_indicator_none_lookup_lookuptablefindv2_default_value*	
Tin0	*

Tout0	*#
_output_shapes
:?????????{
0Multi_Year_indicator/SparseToDense/default_valueConst*
_output_shapes
: *
dtype0	*
valueB	 R
??????????
"Multi_Year_indicator/SparseToDenseSparseToDense4Multi_Year_indicator/to_sparse_input/indices:index:09Multi_Year_indicator/to_sparse_input/dense_shape:output:0;Multi_Year_indicator/None_Lookup/LookupTableFindV2:values:09Multi_Year_indicator/SparseToDense/default_value:output:0*
T0	*
Tindices0	*'
_output_shapes
:?????????g
"Multi_Year_indicator/one_hot/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??i
$Multi_Year_indicator/one_hot/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    d
"Multi_Year_indicator/one_hot/depthConst*
_output_shapes
: *
dtype0*
value	B :?
Multi_Year_indicator/one_hotOneHot*Multi_Year_indicator/SparseToDense:dense:0+Multi_Year_indicator/one_hot/depth:output:0+Multi_Year_indicator/one_hot/Const:output:0-Multi_Year_indicator/one_hot/Const_1:output:0*
T0*+
_output_shapes
:?????????}
*Multi_Year_indicator/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
??????????
Multi_Year_indicator/SumSum%Multi_Year_indicator/one_hot:output:03Multi_Year_indicator/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:?????????k
Multi_Year_indicator/ShapeShape!Multi_Year_indicator/Sum:output:0*
T0*
_output_shapes
:r
(Multi_Year_indicator/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: t
*Multi_Year_indicator/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:t
*Multi_Year_indicator/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
"Multi_Year_indicator/strided_sliceStridedSlice#Multi_Year_indicator/Shape:output:01Multi_Year_indicator/strided_slice/stack:output:03Multi_Year_indicator/strided_slice/stack_1:output:03Multi_Year_indicator/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
$Multi_Year_indicator/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
"Multi_Year_indicator/Reshape/shapePack+Multi_Year_indicator/strided_slice:output:0-Multi_Year_indicator/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
Multi_Year_indicator/ReshapeReshape!Multi_Year_indicator/Sum:output:0+Multi_Year_indicator/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????W
N_Patent_App/ShapeShapefeatures_n_patent_app*
T0*
_output_shapes
:j
 N_Patent_App/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: l
"N_Patent_App/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:l
"N_Patent_App/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
N_Patent_App/strided_sliceStridedSliceN_Patent_App/Shape:output:0)N_Patent_App/strided_slice/stack:output:0+N_Patent_App/strided_slice/stack_1:output:0+N_Patent_App/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask^
N_Patent_App/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
N_Patent_App/Reshape/shapePack#N_Patent_App/strided_slice:output:0%N_Patent_App/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
N_Patent_App/ReshapeReshapefeatures_n_patent_app#N_Patent_App/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????W
N_Patent_Reg/ShapeShapefeatures_n_patent_reg*
T0*
_output_shapes
:j
 N_Patent_Reg/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: l
"N_Patent_Reg/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:l
"N_Patent_Reg/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
N_Patent_Reg/strided_sliceStridedSliceN_Patent_Reg/Shape:output:0)N_Patent_Reg/strided_slice/stack:output:0+N_Patent_Reg/strided_slice/stack_1:output:0+N_Patent_Reg/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask^
N_Patent_Reg/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
N_Patent_Reg/Reshape/shapePack#N_Patent_Reg/strided_slice:output:0%N_Patent_Reg/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
N_Patent_Reg/ReshapeReshapefeatures_n_patent_reg#N_Patent_Reg/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????c
N_of_Korean_Patent/ShapeShapefeatures_n_of_korean_patent*
T0*
_output_shapes
:p
&N_of_Korean_Patent/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: r
(N_of_Korean_Patent/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(N_of_Korean_Patent/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
 N_of_Korean_Patent/strided_sliceStridedSlice!N_of_Korean_Patent/Shape:output:0/N_of_Korean_Patent/strided_slice/stack:output:01N_of_Korean_Patent/strided_slice/stack_1:output:01N_of_Korean_Patent/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskd
"N_of_Korean_Patent/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
 N_of_Korean_Patent/Reshape/shapePack)N_of_Korean_Patent/strided_slice:output:0+N_of_Korean_Patent/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
N_of_Korean_Patent/ReshapeReshapefeatures_n_of_korean_patent)N_of_Korean_Patent/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????S
N_of_Paper/ShapeShapefeatures_n_of_paper*
T0*
_output_shapes
:h
N_of_Paper/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: j
 N_of_Paper/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:j
 N_of_Paper/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
N_of_Paper/strided_sliceStridedSliceN_of_Paper/Shape:output:0'N_of_Paper/strided_slice/stack:output:0)N_of_Paper/strided_slice/stack_1:output:0)N_of_Paper/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask\
N_of_Paper/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
N_of_Paper/Reshape/shapePack!N_of_Paper/strided_slice:output:0#N_of_Paper/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
N_of_Paper/ReshapeReshapefeatures_n_of_paper!N_of_Paper/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????O
N_of_SCI/ShapeShapefeatures_n_of_sci*
T0*
_output_shapes
:f
N_of_SCI/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: h
N_of_SCI/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:h
N_of_SCI/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
N_of_SCI/strided_sliceStridedSliceN_of_SCI/Shape:output:0%N_of_SCI/strided_slice/stack:output:0'N_of_SCI/strided_slice/stack_1:output:0'N_of_SCI/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskZ
N_of_SCI/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
N_of_SCI/Reshape/shapePackN_of_SCI/strided_slice:output:0!N_of_SCI/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
N_of_SCI/ReshapeReshapefeatures_n_of_sciN_of_SCI/Reshape/shape:output:0*
T0*'
_output_shapes
:??????????
<National_Strategy_2_indicator/to_sparse_input/ignore_value/xConst*
_output_shapes
: *
dtype0*
valueB :
??????????
:National_Strategy_2_indicator/to_sparse_input/ignore_valueCastENational_Strategy_2_indicator/to_sparse_input/ignore_value/x:output:0*

DstT0	*

SrcT0*
_output_shapes
: ?
6National_Strategy_2_indicator/to_sparse_input/NotEqualNotEqualfeatures_national_strategy_2>National_Strategy_2_indicator/to_sparse_input/ignore_value:y:0*
T0	*'
_output_shapes
:??????????
5National_Strategy_2_indicator/to_sparse_input/indicesWhere:National_Strategy_2_indicator/to_sparse_input/NotEqual:z:0*'
_output_shapes
:??????????
4National_Strategy_2_indicator/to_sparse_input/valuesGatherNdfeatures_national_strategy_2=National_Strategy_2_indicator/to_sparse_input/indices:index:0*
Tindices0	*
Tparams0	*#
_output_shapes
:??????????
9National_Strategy_2_indicator/to_sparse_input/dense_shapeShapefeatures_national_strategy_2*
T0	*
_output_shapes
:*
out_type0	?
;National_Strategy_2_indicator/None_Lookup/LookupTableFindV2LookupTableFindV2Hnational_strategy_2_indicator_none_lookup_lookuptablefindv2_table_handle=National_Strategy_2_indicator/to_sparse_input/values:output:0Inational_strategy_2_indicator_none_lookup_lookuptablefindv2_default_value*	
Tin0	*

Tout0	*#
_output_shapes
:??????????
9National_Strategy_2_indicator/SparseToDense/default_valueConst*
_output_shapes
: *
dtype0	*
valueB	 R
??????????
+National_Strategy_2_indicator/SparseToDenseSparseToDense=National_Strategy_2_indicator/to_sparse_input/indices:index:0BNational_Strategy_2_indicator/to_sparse_input/dense_shape:output:0DNational_Strategy_2_indicator/None_Lookup/LookupTableFindV2:values:0BNational_Strategy_2_indicator/SparseToDense/default_value:output:0*
T0	*
Tindices0	*'
_output_shapes
:?????????p
+National_Strategy_2_indicator/one_hot/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??r
-National_Strategy_2_indicator/one_hot/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    m
+National_Strategy_2_indicator/one_hot/depthConst*
_output_shapes
: *
dtype0*
value	B :?
%National_Strategy_2_indicator/one_hotOneHot3National_Strategy_2_indicator/SparseToDense:dense:04National_Strategy_2_indicator/one_hot/depth:output:04National_Strategy_2_indicator/one_hot/Const:output:06National_Strategy_2_indicator/one_hot/Const_1:output:0*
T0*+
_output_shapes
:??????????
3National_Strategy_2_indicator/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
??????????
!National_Strategy_2_indicator/SumSum.National_Strategy_2_indicator/one_hot:output:0<National_Strategy_2_indicator/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:?????????}
#National_Strategy_2_indicator/ShapeShape*National_Strategy_2_indicator/Sum:output:0*
T0*
_output_shapes
:{
1National_Strategy_2_indicator/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: }
3National_Strategy_2_indicator/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:}
3National_Strategy_2_indicator/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
+National_Strategy_2_indicator/strided_sliceStridedSlice,National_Strategy_2_indicator/Shape:output:0:National_Strategy_2_indicator/strided_slice/stack:output:0<National_Strategy_2_indicator/strided_slice/stack_1:output:0<National_Strategy_2_indicator/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_masko
-National_Strategy_2_indicator/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
+National_Strategy_2_indicator/Reshape/shapePack4National_Strategy_2_indicator/strided_slice:output:06National_Strategy_2_indicator/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
%National_Strategy_2_indicator/ReshapeReshape*National_Strategy_2_indicator/Sum:output:04National_Strategy_2_indicator/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????{
0RnD_Org_indicator/to_sparse_input/ignore_value/xConst*
_output_shapes
: *
dtype0*
valueB :
??????????
.RnD_Org_indicator/to_sparse_input/ignore_valueCast9RnD_Org_indicator/to_sparse_input/ignore_value/x:output:0*

DstT0	*

SrcT0*
_output_shapes
: ?
*RnD_Org_indicator/to_sparse_input/NotEqualNotEqualfeatures_rnd_org2RnD_Org_indicator/to_sparse_input/ignore_value:y:0*
T0	*'
_output_shapes
:??????????
)RnD_Org_indicator/to_sparse_input/indicesWhere.RnD_Org_indicator/to_sparse_input/NotEqual:z:0*'
_output_shapes
:??????????
(RnD_Org_indicator/to_sparse_input/valuesGatherNdfeatures_rnd_org1RnD_Org_indicator/to_sparse_input/indices:index:0*
Tindices0	*
Tparams0	*#
_output_shapes
:?????????}
-RnD_Org_indicator/to_sparse_input/dense_shapeShapefeatures_rnd_org*
T0	*
_output_shapes
:*
out_type0	?
/RnD_Org_indicator/None_Lookup/LookupTableFindV2LookupTableFindV2<rnd_org_indicator_none_lookup_lookuptablefindv2_table_handle1RnD_Org_indicator/to_sparse_input/values:output:0=rnd_org_indicator_none_lookup_lookuptablefindv2_default_value*	
Tin0	*

Tout0	*#
_output_shapes
:?????????x
-RnD_Org_indicator/SparseToDense/default_valueConst*
_output_shapes
: *
dtype0	*
valueB	 R
??????????
RnD_Org_indicator/SparseToDenseSparseToDense1RnD_Org_indicator/to_sparse_input/indices:index:06RnD_Org_indicator/to_sparse_input/dense_shape:output:08RnD_Org_indicator/None_Lookup/LookupTableFindV2:values:06RnD_Org_indicator/SparseToDense/default_value:output:0*
T0	*
Tindices0	*'
_output_shapes
:?????????d
RnD_Org_indicator/one_hot/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??f
!RnD_Org_indicator/one_hot/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    a
RnD_Org_indicator/one_hot/depthConst*
_output_shapes
: *
dtype0*
value	B :?
RnD_Org_indicator/one_hotOneHot'RnD_Org_indicator/SparseToDense:dense:0(RnD_Org_indicator/one_hot/depth:output:0(RnD_Org_indicator/one_hot/Const:output:0*RnD_Org_indicator/one_hot/Const_1:output:0*
T0*+
_output_shapes
:?????????z
'RnD_Org_indicator/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
??????????
RnD_Org_indicator/SumSum"RnD_Org_indicator/one_hot:output:00RnD_Org_indicator/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:?????????e
RnD_Org_indicator/ShapeShapeRnD_Org_indicator/Sum:output:0*
T0*
_output_shapes
:o
%RnD_Org_indicator/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: q
'RnD_Org_indicator/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:q
'RnD_Org_indicator/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
RnD_Org_indicator/strided_sliceStridedSlice RnD_Org_indicator/Shape:output:0.RnD_Org_indicator/strided_slice/stack:output:00RnD_Org_indicator/strided_slice/stack_1:output:00RnD_Org_indicator/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskc
!RnD_Org_indicator/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
RnD_Org_indicator/Reshape/shapePack(RnD_Org_indicator/strided_slice:output:0*RnD_Org_indicator/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
RnD_Org_indicator/ReshapeReshapeRnD_Org_indicator/Sum:output:0(RnD_Org_indicator/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????}
2RnD_Stage_indicator/to_sparse_input/ignore_value/xConst*
_output_shapes
: *
dtype0*
valueB :
??????????
0RnD_Stage_indicator/to_sparse_input/ignore_valueCast;RnD_Stage_indicator/to_sparse_input/ignore_value/x:output:0*

DstT0	*

SrcT0*
_output_shapes
: ?
,RnD_Stage_indicator/to_sparse_input/NotEqualNotEqualfeatures_rnd_stage4RnD_Stage_indicator/to_sparse_input/ignore_value:y:0*
T0	*'
_output_shapes
:??????????
+RnD_Stage_indicator/to_sparse_input/indicesWhere0RnD_Stage_indicator/to_sparse_input/NotEqual:z:0*'
_output_shapes
:??????????
*RnD_Stage_indicator/to_sparse_input/valuesGatherNdfeatures_rnd_stage3RnD_Stage_indicator/to_sparse_input/indices:index:0*
Tindices0	*
Tparams0	*#
_output_shapes
:??????????
/RnD_Stage_indicator/to_sparse_input/dense_shapeShapefeatures_rnd_stage*
T0	*
_output_shapes
:*
out_type0	?
1RnD_Stage_indicator/None_Lookup/LookupTableFindV2LookupTableFindV2>rnd_stage_indicator_none_lookup_lookuptablefindv2_table_handle3RnD_Stage_indicator/to_sparse_input/values:output:0?rnd_stage_indicator_none_lookup_lookuptablefindv2_default_value*	
Tin0	*

Tout0	*#
_output_shapes
:?????????z
/RnD_Stage_indicator/SparseToDense/default_valueConst*
_output_shapes
: *
dtype0	*
valueB	 R
??????????
!RnD_Stage_indicator/SparseToDenseSparseToDense3RnD_Stage_indicator/to_sparse_input/indices:index:08RnD_Stage_indicator/to_sparse_input/dense_shape:output:0:RnD_Stage_indicator/None_Lookup/LookupTableFindV2:values:08RnD_Stage_indicator/SparseToDense/default_value:output:0*
T0	*
Tindices0	*'
_output_shapes
:?????????f
!RnD_Stage_indicator/one_hot/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??h
#RnD_Stage_indicator/one_hot/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    c
!RnD_Stage_indicator/one_hot/depthConst*
_output_shapes
: *
dtype0*
value	B :?
RnD_Stage_indicator/one_hotOneHot)RnD_Stage_indicator/SparseToDense:dense:0*RnD_Stage_indicator/one_hot/depth:output:0*RnD_Stage_indicator/one_hot/Const:output:0,RnD_Stage_indicator/one_hot/Const_1:output:0*
T0*+
_output_shapes
:?????????|
)RnD_Stage_indicator/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
??????????
RnD_Stage_indicator/SumSum$RnD_Stage_indicator/one_hot:output:02RnD_Stage_indicator/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:?????????i
RnD_Stage_indicator/ShapeShape RnD_Stage_indicator/Sum:output:0*
T0*
_output_shapes
:q
'RnD_Stage_indicator/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: s
)RnD_Stage_indicator/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:s
)RnD_Stage_indicator/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
!RnD_Stage_indicator/strided_sliceStridedSlice"RnD_Stage_indicator/Shape:output:00RnD_Stage_indicator/strided_slice/stack:output:02RnD_Stage_indicator/strided_slice/stack_1:output:02RnD_Stage_indicator/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maske
#RnD_Stage_indicator/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
!RnD_Stage_indicator/Reshape/shapePack*RnD_Stage_indicator/strided_slice:output:0,RnD_Stage_indicator/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
RnD_Stage_indicator/ReshapeReshape RnD_Stage_indicator/Sum:output:0*RnD_Stage_indicator/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????u
4STP_Code_11_indicator/to_sparse_input/ignore_value/xConst*
_output_shapes
: *
dtype0*
valueB B ?
.STP_Code_11_indicator/to_sparse_input/NotEqualNotEqualfeatures_stp_code_11=STP_Code_11_indicator/to_sparse_input/ignore_value/x:output:0*
T0*'
_output_shapes
:??????????
-STP_Code_11_indicator/to_sparse_input/indicesWhere2STP_Code_11_indicator/to_sparse_input/NotEqual:z:0*'
_output_shapes
:??????????
,STP_Code_11_indicator/to_sparse_input/valuesGatherNdfeatures_stp_code_115STP_Code_11_indicator/to_sparse_input/indices:index:0*
Tindices0	*
Tparams0*#
_output_shapes
:??????????
1STP_Code_11_indicator/to_sparse_input/dense_shapeShapefeatures_stp_code_11*
T0*
_output_shapes
:*
out_type0	?
3STP_Code_11_indicator/None_Lookup/LookupTableFindV2LookupTableFindV2@stp_code_11_indicator_none_lookup_lookuptablefindv2_table_handle5STP_Code_11_indicator/to_sparse_input/values:output:0Astp_code_11_indicator_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:?????????|
1STP_Code_11_indicator/SparseToDense/default_valueConst*
_output_shapes
: *
dtype0	*
valueB	 R
??????????
#STP_Code_11_indicator/SparseToDenseSparseToDense5STP_Code_11_indicator/to_sparse_input/indices:index:0:STP_Code_11_indicator/to_sparse_input/dense_shape:output:0<STP_Code_11_indicator/None_Lookup/LookupTableFindV2:values:0:STP_Code_11_indicator/SparseToDense/default_value:output:0*
T0	*
Tindices0	*'
_output_shapes
:?????????h
#STP_Code_11_indicator/one_hot/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??j
%STP_Code_11_indicator/one_hot/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    f
#STP_Code_11_indicator/one_hot/depthConst*
_output_shapes
: *
dtype0*
value
B :??
STP_Code_11_indicator/one_hotOneHot+STP_Code_11_indicator/SparseToDense:dense:0,STP_Code_11_indicator/one_hot/depth:output:0,STP_Code_11_indicator/one_hot/Const:output:0.STP_Code_11_indicator/one_hot/Const_1:output:0*
T0*,
_output_shapes
:??????????~
+STP_Code_11_indicator/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
??????????
STP_Code_11_indicator/SumSum&STP_Code_11_indicator/one_hot:output:04STP_Code_11_indicator/Sum/reduction_indices:output:0*
T0*(
_output_shapes
:??????????m
STP_Code_11_indicator/ShapeShape"STP_Code_11_indicator/Sum:output:0*
T0*
_output_shapes
:s
)STP_Code_11_indicator/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: u
+STP_Code_11_indicator/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+STP_Code_11_indicator/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
#STP_Code_11_indicator/strided_sliceStridedSlice$STP_Code_11_indicator/Shape:output:02STP_Code_11_indicator/strided_slice/stack:output:04STP_Code_11_indicator/strided_slice/stack_1:output:04STP_Code_11_indicator/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskh
%STP_Code_11_indicator/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value
B :??
#STP_Code_11_indicator/Reshape/shapePack,STP_Code_11_indicator/strided_slice:output:0.STP_Code_11_indicator/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
STP_Code_11_indicator/ReshapeReshape"STP_Code_11_indicator/Sum:output:0,STP_Code_11_indicator/Reshape/shape:output:0*
T0*(
_output_shapes
:??????????a
STP_Code_1_Weight/ShapeShapefeatures_stp_code_1_weight*
T0*
_output_shapes
:o
%STP_Code_1_Weight/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: q
'STP_Code_1_Weight/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:q
'STP_Code_1_Weight/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
STP_Code_1_Weight/strided_sliceStridedSlice STP_Code_1_Weight/Shape:output:0.STP_Code_1_Weight/strided_slice/stack:output:00STP_Code_1_Weight/strided_slice/stack_1:output:00STP_Code_1_Weight/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskc
!STP_Code_1_Weight/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
STP_Code_1_Weight/Reshape/shapePack(STP_Code_1_Weight/strided_slice:output:0*STP_Code_1_Weight/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
STP_Code_1_Weight/ReshapeReshapefeatures_stp_code_1_weight(STP_Code_1_Weight/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????u
4STP_Code_21_indicator/to_sparse_input/ignore_value/xConst*
_output_shapes
: *
dtype0*
valueB B ?
.STP_Code_21_indicator/to_sparse_input/NotEqualNotEqualfeatures_stp_code_21=STP_Code_21_indicator/to_sparse_input/ignore_value/x:output:0*
T0*'
_output_shapes
:??????????
-STP_Code_21_indicator/to_sparse_input/indicesWhere2STP_Code_21_indicator/to_sparse_input/NotEqual:z:0*'
_output_shapes
:??????????
,STP_Code_21_indicator/to_sparse_input/valuesGatherNdfeatures_stp_code_215STP_Code_21_indicator/to_sparse_input/indices:index:0*
Tindices0	*
Tparams0*#
_output_shapes
:??????????
1STP_Code_21_indicator/to_sparse_input/dense_shapeShapefeatures_stp_code_21*
T0*
_output_shapes
:*
out_type0	?
3STP_Code_21_indicator/None_Lookup/LookupTableFindV2LookupTableFindV2@stp_code_21_indicator_none_lookup_lookuptablefindv2_table_handle5STP_Code_21_indicator/to_sparse_input/values:output:0Astp_code_21_indicator_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:?????????|
1STP_Code_21_indicator/SparseToDense/default_valueConst*
_output_shapes
: *
dtype0	*
valueB	 R
??????????
#STP_Code_21_indicator/SparseToDenseSparseToDense5STP_Code_21_indicator/to_sparse_input/indices:index:0:STP_Code_21_indicator/to_sparse_input/dense_shape:output:0<STP_Code_21_indicator/None_Lookup/LookupTableFindV2:values:0:STP_Code_21_indicator/SparseToDense/default_value:output:0*
T0	*
Tindices0	*'
_output_shapes
:?????????h
#STP_Code_21_indicator/one_hot/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??j
%STP_Code_21_indicator/one_hot/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    f
#STP_Code_21_indicator/one_hot/depthConst*
_output_shapes
: *
dtype0*
value
B :??
STP_Code_21_indicator/one_hotOneHot+STP_Code_21_indicator/SparseToDense:dense:0,STP_Code_21_indicator/one_hot/depth:output:0,STP_Code_21_indicator/one_hot/Const:output:0.STP_Code_21_indicator/one_hot/Const_1:output:0*
T0*,
_output_shapes
:??????????~
+STP_Code_21_indicator/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
??????????
STP_Code_21_indicator/SumSum&STP_Code_21_indicator/one_hot:output:04STP_Code_21_indicator/Sum/reduction_indices:output:0*
T0*(
_output_shapes
:??????????m
STP_Code_21_indicator/ShapeShape"STP_Code_21_indicator/Sum:output:0*
T0*
_output_shapes
:s
)STP_Code_21_indicator/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: u
+STP_Code_21_indicator/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+STP_Code_21_indicator/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
#STP_Code_21_indicator/strided_sliceStridedSlice$STP_Code_21_indicator/Shape:output:02STP_Code_21_indicator/strided_slice/stack:output:04STP_Code_21_indicator/strided_slice/stack_1:output:04STP_Code_21_indicator/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskh
%STP_Code_21_indicator/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value
B :??
#STP_Code_21_indicator/Reshape/shapePack,STP_Code_21_indicator/strided_slice:output:0.STP_Code_21_indicator/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
STP_Code_21_indicator/ReshapeReshape"STP_Code_21_indicator/Sum:output:0,STP_Code_21_indicator/Reshape/shape:output:0*
T0*(
_output_shapes
:??????????a
STP_Code_2_Weight/ShapeShapefeatures_stp_code_2_weight*
T0*
_output_shapes
:o
%STP_Code_2_Weight/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: q
'STP_Code_2_Weight/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:q
'STP_Code_2_Weight/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
STP_Code_2_Weight/strided_sliceStridedSlice STP_Code_2_Weight/Shape:output:0.STP_Code_2_Weight/strided_slice/stack:output:00STP_Code_2_Weight/strided_slice/stack_1:output:00STP_Code_2_Weight/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskc
!STP_Code_2_Weight/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
STP_Code_2_Weight/Reshape/shapePack(STP_Code_2_Weight/strided_slice:output:0*STP_Code_2_Weight/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
STP_Code_2_Weight/ReshapeReshapefeatures_stp_code_2_weight(STP_Code_2_Weight/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????z
/SixT_2_indicator/to_sparse_input/ignore_value/xConst*
_output_shapes
: *
dtype0*
valueB :
??????????
-SixT_2_indicator/to_sparse_input/ignore_valueCast8SixT_2_indicator/to_sparse_input/ignore_value/x:output:0*

DstT0	*

SrcT0*
_output_shapes
: ?
)SixT_2_indicator/to_sparse_input/NotEqualNotEqualfeatures_sixt_21SixT_2_indicator/to_sparse_input/ignore_value:y:0*
T0	*'
_output_shapes
:??????????
(SixT_2_indicator/to_sparse_input/indicesWhere-SixT_2_indicator/to_sparse_input/NotEqual:z:0*'
_output_shapes
:??????????
'SixT_2_indicator/to_sparse_input/valuesGatherNdfeatures_sixt_20SixT_2_indicator/to_sparse_input/indices:index:0*
Tindices0	*
Tparams0	*#
_output_shapes
:?????????{
,SixT_2_indicator/to_sparse_input/dense_shapeShapefeatures_sixt_2*
T0	*
_output_shapes
:*
out_type0	?
.SixT_2_indicator/None_Lookup/LookupTableFindV2LookupTableFindV2;sixt_2_indicator_none_lookup_lookuptablefindv2_table_handle0SixT_2_indicator/to_sparse_input/values:output:0<sixt_2_indicator_none_lookup_lookuptablefindv2_default_value*	
Tin0	*

Tout0	*#
_output_shapes
:?????????w
,SixT_2_indicator/SparseToDense/default_valueConst*
_output_shapes
: *
dtype0	*
valueB	 R
??????????
SixT_2_indicator/SparseToDenseSparseToDense0SixT_2_indicator/to_sparse_input/indices:index:05SixT_2_indicator/to_sparse_input/dense_shape:output:07SixT_2_indicator/None_Lookup/LookupTableFindV2:values:05SixT_2_indicator/SparseToDense/default_value:output:0*
T0	*
Tindices0	*'
_output_shapes
:?????????c
SixT_2_indicator/one_hot/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??e
 SixT_2_indicator/one_hot/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    `
SixT_2_indicator/one_hot/depthConst*
_output_shapes
: *
dtype0*
value	B :?
SixT_2_indicator/one_hotOneHot&SixT_2_indicator/SparseToDense:dense:0'SixT_2_indicator/one_hot/depth:output:0'SixT_2_indicator/one_hot/Const:output:0)SixT_2_indicator/one_hot/Const_1:output:0*
T0*+
_output_shapes
:?????????y
&SixT_2_indicator/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
??????????
SixT_2_indicator/SumSum!SixT_2_indicator/one_hot:output:0/SixT_2_indicator/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:?????????c
SixT_2_indicator/ShapeShapeSixT_2_indicator/Sum:output:0*
T0*
_output_shapes
:n
$SixT_2_indicator/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: p
&SixT_2_indicator/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:p
&SixT_2_indicator/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
SixT_2_indicator/strided_sliceStridedSliceSixT_2_indicator/Shape:output:0-SixT_2_indicator/strided_slice/stack:output:0/SixT_2_indicator/strided_slice/stack_1:output:0/SixT_2_indicator/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskb
 SixT_2_indicator/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
SixT_2_indicator/Reshape/shapePack'SixT_2_indicator/strided_slice:output:0)SixT_2_indicator/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
SixT_2_indicator/ReshapeReshapeSixT_2_indicator/Sum:output:0'SixT_2_indicator/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????G

Year/ShapeShapefeatures_year*
T0*
_output_shapes
:b
Year/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: d
Year/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:d
Year/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
Year/strided_sliceStridedSliceYear/Shape:output:0!Year/strided_slice/stack:output:0#Year/strided_slice/stack_1:output:0#Year/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskV
Year/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
Year/Reshape/shapePackYear/strided_slice:output:0Year/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:u
Year/ReshapeReshapefeatures_yearYear/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????V
concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
??????????
concatConcatV2*Application_Area_1_Weight/Reshape:output:0-Application_Area_1_indicator/Reshape:output:0*Application_Area_2_Weight/Reshape:output:0-Application_Area_2_indicator/Reshape:output:0(Cowork_Abroad_indicator/Reshape:output:0%Cowork_Cor_indicator/Reshape:output:0&Cowork_Inst_indicator/Reshape:output:0%Cowork_Uni_indicator/Reshape:output:0%Cowork_etc_indicator/Reshape:output:0&Econ_Social_indicator/Reshape:output:0%Green_Tech_indicator/Reshape:output:0Log_Duration/Reshape:output:0Log_RnD_Fund/Reshape:output:0%Multi_Year_indicator/Reshape:output:0N_Patent_App/Reshape:output:0N_Patent_Reg/Reshape:output:0#N_of_Korean_Patent/Reshape:output:0N_of_Paper/Reshape:output:0N_of_SCI/Reshape:output:0.National_Strategy_2_indicator/Reshape:output:0"RnD_Org_indicator/Reshape:output:0$RnD_Stage_indicator/Reshape:output:0&STP_Code_11_indicator/Reshape:output:0"STP_Code_1_Weight/Reshape:output:0&STP_Code_21_indicator/Reshape:output:0"STP_Code_2_Weight/Reshape:output:0!SixT_2_indicator/Reshape:output:0Year/Reshape:output:0concat/axis:output:0*
N*
T0*(
_output_shapes
:??????????_
IdentityIdentityconcat:output:0^NoOp*
T0*(
_output_shapes
:???????????
NoOpNoOp;^Application_Area_1_indicator/None_Lookup/LookupTableFindV2;^Application_Area_2_indicator/None_Lookup/LookupTableFindV26^Cowork_Abroad_indicator/None_Lookup/LookupTableFindV23^Cowork_Cor_indicator/None_Lookup/LookupTableFindV24^Cowork_Inst_indicator/None_Lookup/LookupTableFindV23^Cowork_Uni_indicator/None_Lookup/LookupTableFindV23^Cowork_etc_indicator/None_Lookup/LookupTableFindV24^Econ_Social_indicator/None_Lookup/LookupTableFindV23^Green_Tech_indicator/None_Lookup/LookupTableFindV23^Multi_Year_indicator/None_Lookup/LookupTableFindV2<^National_Strategy_2_indicator/None_Lookup/LookupTableFindV20^RnD_Org_indicator/None_Lookup/LookupTableFindV22^RnD_Stage_indicator/None_Lookup/LookupTableFindV24^STP_Code_11_indicator/None_Lookup/LookupTableFindV24^STP_Code_21_indicator/None_Lookup/LookupTableFindV2/^SixT_2_indicator/None_Lookup/LookupTableFindV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2x
:Application_Area_1_indicator/None_Lookup/LookupTableFindV2:Application_Area_1_indicator/None_Lookup/LookupTableFindV22x
:Application_Area_2_indicator/None_Lookup/LookupTableFindV2:Application_Area_2_indicator/None_Lookup/LookupTableFindV22n
5Cowork_Abroad_indicator/None_Lookup/LookupTableFindV25Cowork_Abroad_indicator/None_Lookup/LookupTableFindV22h
2Cowork_Cor_indicator/None_Lookup/LookupTableFindV22Cowork_Cor_indicator/None_Lookup/LookupTableFindV22j
3Cowork_Inst_indicator/None_Lookup/LookupTableFindV23Cowork_Inst_indicator/None_Lookup/LookupTableFindV22h
2Cowork_Uni_indicator/None_Lookup/LookupTableFindV22Cowork_Uni_indicator/None_Lookup/LookupTableFindV22h
2Cowork_etc_indicator/None_Lookup/LookupTableFindV22Cowork_etc_indicator/None_Lookup/LookupTableFindV22j
3Econ_Social_indicator/None_Lookup/LookupTableFindV23Econ_Social_indicator/None_Lookup/LookupTableFindV22h
2Green_Tech_indicator/None_Lookup/LookupTableFindV22Green_Tech_indicator/None_Lookup/LookupTableFindV22h
2Multi_Year_indicator/None_Lookup/LookupTableFindV22Multi_Year_indicator/None_Lookup/LookupTableFindV22z
;National_Strategy_2_indicator/None_Lookup/LookupTableFindV2;National_Strategy_2_indicator/None_Lookup/LookupTableFindV22b
/RnD_Org_indicator/None_Lookup/LookupTableFindV2/RnD_Org_indicator/None_Lookup/LookupTableFindV22f
1RnD_Stage_indicator/None_Lookup/LookupTableFindV21RnD_Stage_indicator/None_Lookup/LookupTableFindV22j
3STP_Code_11_indicator/None_Lookup/LookupTableFindV23STP_Code_11_indicator/None_Lookup/LookupTableFindV22j
3STP_Code_21_indicator/None_Lookup/LookupTableFindV23STP_Code_21_indicator/None_Lookup/LookupTableFindV22`
.SixT_2_indicator/None_Lookup/LookupTableFindV2.SixT_2_indicator/None_Lookup/LookupTableFindV2:d `
'
_output_shapes
:?????????
5
_user_specified_namefeatures/Application_Area_1:kg
'
_output_shapes
:?????????
<
_user_specified_name$"features/Application_Area_1_Weight:d`
'
_output_shapes
:?????????
5
_user_specified_namefeatures/Application_Area_2:kg
'
_output_shapes
:?????????
<
_user_specified_name$"features/Application_Area_2_Weight:_[
'
_output_shapes
:?????????
0
_user_specified_namefeatures/Cowork_Abroad:\X
'
_output_shapes
:?????????
-
_user_specified_namefeatures/Cowork_Cor:]Y
'
_output_shapes
:?????????
.
_user_specified_namefeatures/Cowork_Inst:\X
'
_output_shapes
:?????????
-
_user_specified_namefeatures/Cowork_Uni:\X
'
_output_shapes
:?????????
-
_user_specified_namefeatures/Cowork_etc:]	Y
'
_output_shapes
:?????????
.
_user_specified_namefeatures/Econ_Social:\
X
'
_output_shapes
:?????????
-
_user_specified_namefeatures/Green_Tech:^Z
'
_output_shapes
:?????????
/
_user_specified_namefeatures/Log_Duration:^Z
'
_output_shapes
:?????????
/
_user_specified_namefeatures/Log_RnD_Fund:\X
'
_output_shapes
:?????????
-
_user_specified_namefeatures/Multi_Year:^Z
'
_output_shapes
:?????????
/
_user_specified_namefeatures/N_Patent_App:^Z
'
_output_shapes
:?????????
/
_user_specified_namefeatures/N_Patent_Reg:d`
'
_output_shapes
:?????????
5
_user_specified_namefeatures/N_of_Korean_Patent:\X
'
_output_shapes
:?????????
-
_user_specified_namefeatures/N_of_Paper:ZV
'
_output_shapes
:?????????
+
_user_specified_namefeatures/N_of_SCI:ea
'
_output_shapes
:?????????
6
_user_specified_namefeatures/National_Strategy_2:YU
'
_output_shapes
:?????????
*
_user_specified_namefeatures/RnD_Org:[W
'
_output_shapes
:?????????
,
_user_specified_namefeatures/RnD_Stage:]Y
'
_output_shapes
:?????????
.
_user_specified_namefeatures/STP_Code_11:c_
'
_output_shapes
:?????????
4
_user_specified_namefeatures/STP_Code_1_Weight:]Y
'
_output_shapes
:?????????
.
_user_specified_namefeatures/STP_Code_21:c_
'
_output_shapes
:?????????
4
_user_specified_namefeatures/STP_Code_2_Weight:XT
'
_output_shapes
:?????????
)
_user_specified_namefeatures/SixT_2:VR
'
_output_shapes
:?????????
'
_user_specified_namefeatures/Year:

_output_shapes
: :

_output_shapes
: :!

_output_shapes
: :#

_output_shapes
: :%

_output_shapes
: :'

_output_shapes
: :)

_output_shapes
: :+

_output_shapes
: :-

_output_shapes
: :/

_output_shapes
: :1

_output_shapes
: :3

_output_shapes
: :5

_output_shapes
: :7

_output_shapes
: :9

_output_shapes
: :;

_output_shapes
: 
?
-
__inference__destroyer_133794
identityG
ConstConst*
_output_shapes
: *
dtype0*
value	B :E
IdentityIdentityConst:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 
?

?
A__inference_dense_layer_call_and_return_conditional_losses_133509

inputs2
matmul_readvariableop_resource:
??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????Q
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:??????????b
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:??????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
-
__inference__destroyer_133686
identityG
ConstConst*
_output_shapes
: *
dtype0*
value	B :E
IdentityIdentityConst:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 
?
?
__inference__initializer_1336632
.table_init299_lookuptableimportv2_table_handle*
&table_init299_lookuptableimportv2_keys,
(table_init299_lookuptableimportv2_values	
identity??!table_init299/LookupTableImportV2?
!table_init299/LookupTableImportV2LookupTableImportV2.table_init299_lookuptableimportv2_table_handle&table_init299_lookuptableimportv2_keys(table_init299_lookuptableimportv2_values*	
Tin0*

Tout0	*
_output_shapes
 G
ConstConst*
_output_shapes
: *
dtype0*
value	B :L
IdentityIdentityConst:output:0^NoOp*
T0*
_output_shapes
: j
NoOpNoOp"^table_init299/LookupTableImportV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*!
_input_shapes
: :!:!2F
!table_init299/LookupTableImportV2!table_init299/LookupTableImportV2: 

_output_shapes
:!: 

_output_shapes
:!
?
-
__inference__destroyer_133884
identityG
ConstConst*
_output_shapes
: *
dtype0*
value	B :E
IdentityIdentityConst:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 
?
;
__inference__creator_133709
identity??
hash_tablek

hash_tableHashTableV2*
_output_shapes
: *
	key_dtype0*
shared_name410*
value_dtype0	W
IdentityIdentityhash_table:table_handle:0^NoOp*
T0*
_output_shapes
: S
NoOpNoOp^hash_table*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 2

hash_table
hash_table
?
-
__inference__destroyer_133848
identityG
ConstConst*
_output_shapes
: *
dtype0*
value	B :E
IdentityIdentityConst:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 
?
-
__inference__destroyer_133902
identityG
ConstConst*
_output_shapes
: *
dtype0*
value	B :E
IdentityIdentityConst:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 
??
?*
F__inference_sequential_layer_call_and_return_conditional_losses_131685
inputs_application_area_1$
 inputs_application_area_1_weight
inputs_application_area_2$
 inputs_application_area_2_weight
inputs_cowork_abroad
inputs_cowork_cor
inputs_cowork_inst
inputs_cowork_uni
inputs_cowork_etc
inputs_econ_social	
inputs_green_tech	
inputs_log_duration
inputs_log_rnd_fund
inputs_multi_year	
inputs_n_patent_app
inputs_n_patent_reg
inputs_n_of_korean_patent
inputs_n_of_paper
inputs_n_of_sci
inputs_national_strategy_2	
inputs_rnd_org	
inputs_rnd_stage	
inputs_stp_code_11
inputs_stp_code_1_weight
inputs_stp_code_21
inputs_stp_code_2_weight
inputs_sixt_2	
inputs_yearZ
Vdense_features_application_area_1_indicator_none_lookup_lookuptablefindv2_table_handle[
Wdense_features_application_area_1_indicator_none_lookup_lookuptablefindv2_default_value	Z
Vdense_features_application_area_2_indicator_none_lookup_lookuptablefindv2_table_handle[
Wdense_features_application_area_2_indicator_none_lookup_lookuptablefindv2_default_value	U
Qdense_features_cowork_abroad_indicator_none_lookup_lookuptablefindv2_table_handleV
Rdense_features_cowork_abroad_indicator_none_lookup_lookuptablefindv2_default_value	R
Ndense_features_cowork_cor_indicator_none_lookup_lookuptablefindv2_table_handleS
Odense_features_cowork_cor_indicator_none_lookup_lookuptablefindv2_default_value	S
Odense_features_cowork_inst_indicator_none_lookup_lookuptablefindv2_table_handleT
Pdense_features_cowork_inst_indicator_none_lookup_lookuptablefindv2_default_value	R
Ndense_features_cowork_uni_indicator_none_lookup_lookuptablefindv2_table_handleS
Odense_features_cowork_uni_indicator_none_lookup_lookuptablefindv2_default_value	R
Ndense_features_cowork_etc_indicator_none_lookup_lookuptablefindv2_table_handleS
Odense_features_cowork_etc_indicator_none_lookup_lookuptablefindv2_default_value	S
Odense_features_econ_social_indicator_none_lookup_lookuptablefindv2_table_handleT
Pdense_features_econ_social_indicator_none_lookup_lookuptablefindv2_default_value	R
Ndense_features_green_tech_indicator_none_lookup_lookuptablefindv2_table_handleS
Odense_features_green_tech_indicator_none_lookup_lookuptablefindv2_default_value	R
Ndense_features_multi_year_indicator_none_lookup_lookuptablefindv2_table_handleS
Odense_features_multi_year_indicator_none_lookup_lookuptablefindv2_default_value	[
Wdense_features_national_strategy_2_indicator_none_lookup_lookuptablefindv2_table_handle\
Xdense_features_national_strategy_2_indicator_none_lookup_lookuptablefindv2_default_value	O
Kdense_features_rnd_org_indicator_none_lookup_lookuptablefindv2_table_handleP
Ldense_features_rnd_org_indicator_none_lookup_lookuptablefindv2_default_value	Q
Mdense_features_rnd_stage_indicator_none_lookup_lookuptablefindv2_table_handleR
Ndense_features_rnd_stage_indicator_none_lookup_lookuptablefindv2_default_value	S
Odense_features_stp_code_11_indicator_none_lookup_lookuptablefindv2_table_handleT
Pdense_features_stp_code_11_indicator_none_lookup_lookuptablefindv2_default_value	S
Odense_features_stp_code_21_indicator_none_lookup_lookuptablefindv2_table_handleT
Pdense_features_stp_code_21_indicator_none_lookup_lookuptablefindv2_default_value	N
Jdense_features_sixt_2_indicator_none_lookup_lookuptablefindv2_table_handleO
Kdense_features_sixt_2_indicator_none_lookup_lookuptablefindv2_default_value	8
$dense_matmul_readvariableop_resource:
??4
%dense_biasadd_readvariableop_resource:	?:
&dense_1_matmul_readvariableop_resource:
??6
'dense_1_biasadd_readvariableop_resource:	?:
&dense_2_matmul_readvariableop_resource:
??6
'dense_2_biasadd_readvariableop_resource:	?9
&dense_3_matmul_readvariableop_resource:	?5
'dense_3_biasadd_readvariableop_resource:
identity??dense/BiasAdd/ReadVariableOp?dense/MatMul/ReadVariableOp?dense_1/BiasAdd/ReadVariableOp?dense_1/MatMul/ReadVariableOp?dense_2/BiasAdd/ReadVariableOp?dense_2/MatMul/ReadVariableOp?dense_3/BiasAdd/ReadVariableOp?dense_3/MatMul/ReadVariableOp?Idense_features/Application_Area_1_indicator/None_Lookup/LookupTableFindV2?Idense_features/Application_Area_2_indicator/None_Lookup/LookupTableFindV2?Ddense_features/Cowork_Abroad_indicator/None_Lookup/LookupTableFindV2?Adense_features/Cowork_Cor_indicator/None_Lookup/LookupTableFindV2?Bdense_features/Cowork_Inst_indicator/None_Lookup/LookupTableFindV2?Adense_features/Cowork_Uni_indicator/None_Lookup/LookupTableFindV2?Adense_features/Cowork_etc_indicator/None_Lookup/LookupTableFindV2?Bdense_features/Econ_Social_indicator/None_Lookup/LookupTableFindV2?Adense_features/Green_Tech_indicator/None_Lookup/LookupTableFindV2?Adense_features/Multi_Year_indicator/None_Lookup/LookupTableFindV2?Jdense_features/National_Strategy_2_indicator/None_Lookup/LookupTableFindV2?>dense_features/RnD_Org_indicator/None_Lookup/LookupTableFindV2?@dense_features/RnD_Stage_indicator/None_Lookup/LookupTableFindV2?Bdense_features/STP_Code_11_indicator/None_Lookup/LookupTableFindV2?Bdense_features/STP_Code_21_indicator/None_Lookup/LookupTableFindV2?=dense_features/SixT_2_indicator/None_Lookup/LookupTableFindV2~
.dense_features/Application_Area_1_Weight/ShapeShape inputs_application_area_1_weight*
T0*
_output_shapes
:?
<dense_features/Application_Area_1_Weight/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
>dense_features/Application_Area_1_Weight/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
>dense_features/Application_Area_1_Weight/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
6dense_features/Application_Area_1_Weight/strided_sliceStridedSlice7dense_features/Application_Area_1_Weight/Shape:output:0Edense_features/Application_Area_1_Weight/strided_slice/stack:output:0Gdense_features/Application_Area_1_Weight/strided_slice/stack_1:output:0Gdense_features/Application_Area_1_Weight/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskz
8dense_features/Application_Area_1_Weight/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
6dense_features/Application_Area_1_Weight/Reshape/shapePack?dense_features/Application_Area_1_Weight/strided_slice:output:0Adense_features/Application_Area_1_Weight/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
0dense_features/Application_Area_1_Weight/ReshapeReshape inputs_application_area_1_weight?dense_features/Application_Area_1_Weight/Reshape/shape:output:0*
T0*'
_output_shapes
:??????????
Jdense_features/Application_Area_1_indicator/to_sparse_input/ignore_value/xConst*
_output_shapes
: *
dtype0*
valueB B ?
Ddense_features/Application_Area_1_indicator/to_sparse_input/NotEqualNotEqualinputs_application_area_1Sdense_features/Application_Area_1_indicator/to_sparse_input/ignore_value/x:output:0*
T0*'
_output_shapes
:??????????
Cdense_features/Application_Area_1_indicator/to_sparse_input/indicesWhereHdense_features/Application_Area_1_indicator/to_sparse_input/NotEqual:z:0*'
_output_shapes
:??????????
Bdense_features/Application_Area_1_indicator/to_sparse_input/valuesGatherNdinputs_application_area_1Kdense_features/Application_Area_1_indicator/to_sparse_input/indices:index:0*
Tindices0	*
Tparams0*#
_output_shapes
:??????????
Gdense_features/Application_Area_1_indicator/to_sparse_input/dense_shapeShapeinputs_application_area_1*
T0*
_output_shapes
:*
out_type0	?
Idense_features/Application_Area_1_indicator/None_Lookup/LookupTableFindV2LookupTableFindV2Vdense_features_application_area_1_indicator_none_lookup_lookuptablefindv2_table_handleKdense_features/Application_Area_1_indicator/to_sparse_input/values:output:0Wdense_features_application_area_1_indicator_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:??????????
Gdense_features/Application_Area_1_indicator/SparseToDense/default_valueConst*
_output_shapes
: *
dtype0	*
valueB	 R
??????????
9dense_features/Application_Area_1_indicator/SparseToDenseSparseToDenseKdense_features/Application_Area_1_indicator/to_sparse_input/indices:index:0Pdense_features/Application_Area_1_indicator/to_sparse_input/dense_shape:output:0Rdense_features/Application_Area_1_indicator/None_Lookup/LookupTableFindV2:values:0Pdense_features/Application_Area_1_indicator/SparseToDense/default_value:output:0*
T0	*
Tindices0	*'
_output_shapes
:?????????~
9dense_features/Application_Area_1_indicator/one_hot/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
;dense_features/Application_Area_1_indicator/one_hot/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    {
9dense_features/Application_Area_1_indicator/one_hot/depthConst*
_output_shapes
: *
dtype0*
value	B :!?
3dense_features/Application_Area_1_indicator/one_hotOneHotAdense_features/Application_Area_1_indicator/SparseToDense:dense:0Bdense_features/Application_Area_1_indicator/one_hot/depth:output:0Bdense_features/Application_Area_1_indicator/one_hot/Const:output:0Ddense_features/Application_Area_1_indicator/one_hot/Const_1:output:0*
T0*+
_output_shapes
:?????????!?
Adense_features/Application_Area_1_indicator/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
??????????
/dense_features/Application_Area_1_indicator/SumSum<dense_features/Application_Area_1_indicator/one_hot:output:0Jdense_features/Application_Area_1_indicator/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:?????????!?
1dense_features/Application_Area_1_indicator/ShapeShape8dense_features/Application_Area_1_indicator/Sum:output:0*
T0*
_output_shapes
:?
?dense_features/Application_Area_1_indicator/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
Adense_features/Application_Area_1_indicator/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
Adense_features/Application_Area_1_indicator/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
9dense_features/Application_Area_1_indicator/strided_sliceStridedSlice:dense_features/Application_Area_1_indicator/Shape:output:0Hdense_features/Application_Area_1_indicator/strided_slice/stack:output:0Jdense_features/Application_Area_1_indicator/strided_slice/stack_1:output:0Jdense_features/Application_Area_1_indicator/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask}
;dense_features/Application_Area_1_indicator/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :!?
9dense_features/Application_Area_1_indicator/Reshape/shapePackBdense_features/Application_Area_1_indicator/strided_slice:output:0Ddense_features/Application_Area_1_indicator/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
3dense_features/Application_Area_1_indicator/ReshapeReshape8dense_features/Application_Area_1_indicator/Sum:output:0Bdense_features/Application_Area_1_indicator/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????!~
.dense_features/Application_Area_2_Weight/ShapeShape inputs_application_area_2_weight*
T0*
_output_shapes
:?
<dense_features/Application_Area_2_Weight/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
>dense_features/Application_Area_2_Weight/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
>dense_features/Application_Area_2_Weight/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
6dense_features/Application_Area_2_Weight/strided_sliceStridedSlice7dense_features/Application_Area_2_Weight/Shape:output:0Edense_features/Application_Area_2_Weight/strided_slice/stack:output:0Gdense_features/Application_Area_2_Weight/strided_slice/stack_1:output:0Gdense_features/Application_Area_2_Weight/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskz
8dense_features/Application_Area_2_Weight/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
6dense_features/Application_Area_2_Weight/Reshape/shapePack?dense_features/Application_Area_2_Weight/strided_slice:output:0Adense_features/Application_Area_2_Weight/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
0dense_features/Application_Area_2_Weight/ReshapeReshape inputs_application_area_2_weight?dense_features/Application_Area_2_Weight/Reshape/shape:output:0*
T0*'
_output_shapes
:??????????
Jdense_features/Application_Area_2_indicator/to_sparse_input/ignore_value/xConst*
_output_shapes
: *
dtype0*
valueB B ?
Ddense_features/Application_Area_2_indicator/to_sparse_input/NotEqualNotEqualinputs_application_area_2Sdense_features/Application_Area_2_indicator/to_sparse_input/ignore_value/x:output:0*
T0*'
_output_shapes
:??????????
Cdense_features/Application_Area_2_indicator/to_sparse_input/indicesWhereHdense_features/Application_Area_2_indicator/to_sparse_input/NotEqual:z:0*'
_output_shapes
:??????????
Bdense_features/Application_Area_2_indicator/to_sparse_input/valuesGatherNdinputs_application_area_2Kdense_features/Application_Area_2_indicator/to_sparse_input/indices:index:0*
Tindices0	*
Tparams0*#
_output_shapes
:??????????
Gdense_features/Application_Area_2_indicator/to_sparse_input/dense_shapeShapeinputs_application_area_2*
T0*
_output_shapes
:*
out_type0	?
Idense_features/Application_Area_2_indicator/None_Lookup/LookupTableFindV2LookupTableFindV2Vdense_features_application_area_2_indicator_none_lookup_lookuptablefindv2_table_handleKdense_features/Application_Area_2_indicator/to_sparse_input/values:output:0Wdense_features_application_area_2_indicator_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:??????????
Gdense_features/Application_Area_2_indicator/SparseToDense/default_valueConst*
_output_shapes
: *
dtype0	*
valueB	 R
??????????
9dense_features/Application_Area_2_indicator/SparseToDenseSparseToDenseKdense_features/Application_Area_2_indicator/to_sparse_input/indices:index:0Pdense_features/Application_Area_2_indicator/to_sparse_input/dense_shape:output:0Rdense_features/Application_Area_2_indicator/None_Lookup/LookupTableFindV2:values:0Pdense_features/Application_Area_2_indicator/SparseToDense/default_value:output:0*
T0	*
Tindices0	*'
_output_shapes
:?????????~
9dense_features/Application_Area_2_indicator/one_hot/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
;dense_features/Application_Area_2_indicator/one_hot/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    {
9dense_features/Application_Area_2_indicator/one_hot/depthConst*
_output_shapes
: *
dtype0*
value	B :"?
3dense_features/Application_Area_2_indicator/one_hotOneHotAdense_features/Application_Area_2_indicator/SparseToDense:dense:0Bdense_features/Application_Area_2_indicator/one_hot/depth:output:0Bdense_features/Application_Area_2_indicator/one_hot/Const:output:0Ddense_features/Application_Area_2_indicator/one_hot/Const_1:output:0*
T0*+
_output_shapes
:?????????"?
Adense_features/Application_Area_2_indicator/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
??????????
/dense_features/Application_Area_2_indicator/SumSum<dense_features/Application_Area_2_indicator/one_hot:output:0Jdense_features/Application_Area_2_indicator/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:?????????"?
1dense_features/Application_Area_2_indicator/ShapeShape8dense_features/Application_Area_2_indicator/Sum:output:0*
T0*
_output_shapes
:?
?dense_features/Application_Area_2_indicator/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
Adense_features/Application_Area_2_indicator/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
Adense_features/Application_Area_2_indicator/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
9dense_features/Application_Area_2_indicator/strided_sliceStridedSlice:dense_features/Application_Area_2_indicator/Shape:output:0Hdense_features/Application_Area_2_indicator/strided_slice/stack:output:0Jdense_features/Application_Area_2_indicator/strided_slice/stack_1:output:0Jdense_features/Application_Area_2_indicator/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask}
;dense_features/Application_Area_2_indicator/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :"?
9dense_features/Application_Area_2_indicator/Reshape/shapePackBdense_features/Application_Area_2_indicator/strided_slice:output:0Ddense_features/Application_Area_2_indicator/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
3dense_features/Application_Area_2_indicator/ReshapeReshape8dense_features/Application_Area_2_indicator/Sum:output:0Bdense_features/Application_Area_2_indicator/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????"?
Edense_features/Cowork_Abroad_indicator/to_sparse_input/ignore_value/xConst*
_output_shapes
: *
dtype0*
valueB B ?
?dense_features/Cowork_Abroad_indicator/to_sparse_input/NotEqualNotEqualinputs_cowork_abroadNdense_features/Cowork_Abroad_indicator/to_sparse_input/ignore_value/x:output:0*
T0*'
_output_shapes
:??????????
>dense_features/Cowork_Abroad_indicator/to_sparse_input/indicesWhereCdense_features/Cowork_Abroad_indicator/to_sparse_input/NotEqual:z:0*'
_output_shapes
:??????????
=dense_features/Cowork_Abroad_indicator/to_sparse_input/valuesGatherNdinputs_cowork_abroadFdense_features/Cowork_Abroad_indicator/to_sparse_input/indices:index:0*
Tindices0	*
Tparams0*#
_output_shapes
:??????????
Bdense_features/Cowork_Abroad_indicator/to_sparse_input/dense_shapeShapeinputs_cowork_abroad*
T0*
_output_shapes
:*
out_type0	?
Ddense_features/Cowork_Abroad_indicator/None_Lookup/LookupTableFindV2LookupTableFindV2Qdense_features_cowork_abroad_indicator_none_lookup_lookuptablefindv2_table_handleFdense_features/Cowork_Abroad_indicator/to_sparse_input/values:output:0Rdense_features_cowork_abroad_indicator_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:??????????
Bdense_features/Cowork_Abroad_indicator/SparseToDense/default_valueConst*
_output_shapes
: *
dtype0	*
valueB	 R
??????????
4dense_features/Cowork_Abroad_indicator/SparseToDenseSparseToDenseFdense_features/Cowork_Abroad_indicator/to_sparse_input/indices:index:0Kdense_features/Cowork_Abroad_indicator/to_sparse_input/dense_shape:output:0Mdense_features/Cowork_Abroad_indicator/None_Lookup/LookupTableFindV2:values:0Kdense_features/Cowork_Abroad_indicator/SparseToDense/default_value:output:0*
T0	*
Tindices0	*'
_output_shapes
:?????????y
4dense_features/Cowork_Abroad_indicator/one_hot/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??{
6dense_features/Cowork_Abroad_indicator/one_hot/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    v
4dense_features/Cowork_Abroad_indicator/one_hot/depthConst*
_output_shapes
: *
dtype0*
value	B :?
.dense_features/Cowork_Abroad_indicator/one_hotOneHot<dense_features/Cowork_Abroad_indicator/SparseToDense:dense:0=dense_features/Cowork_Abroad_indicator/one_hot/depth:output:0=dense_features/Cowork_Abroad_indicator/one_hot/Const:output:0?dense_features/Cowork_Abroad_indicator/one_hot/Const_1:output:0*
T0*+
_output_shapes
:??????????
<dense_features/Cowork_Abroad_indicator/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
??????????
*dense_features/Cowork_Abroad_indicator/SumSum7dense_features/Cowork_Abroad_indicator/one_hot:output:0Edense_features/Cowork_Abroad_indicator/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:??????????
,dense_features/Cowork_Abroad_indicator/ShapeShape3dense_features/Cowork_Abroad_indicator/Sum:output:0*
T0*
_output_shapes
:?
:dense_features/Cowork_Abroad_indicator/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
<dense_features/Cowork_Abroad_indicator/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
<dense_features/Cowork_Abroad_indicator/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
4dense_features/Cowork_Abroad_indicator/strided_sliceStridedSlice5dense_features/Cowork_Abroad_indicator/Shape:output:0Cdense_features/Cowork_Abroad_indicator/strided_slice/stack:output:0Edense_features/Cowork_Abroad_indicator/strided_slice/stack_1:output:0Edense_features/Cowork_Abroad_indicator/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskx
6dense_features/Cowork_Abroad_indicator/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
4dense_features/Cowork_Abroad_indicator/Reshape/shapePack=dense_features/Cowork_Abroad_indicator/strided_slice:output:0?dense_features/Cowork_Abroad_indicator/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
.dense_features/Cowork_Abroad_indicator/ReshapeReshape3dense_features/Cowork_Abroad_indicator/Sum:output:0=dense_features/Cowork_Abroad_indicator/Reshape/shape:output:0*
T0*'
_output_shapes
:??????????
Bdense_features/Cowork_Cor_indicator/to_sparse_input/ignore_value/xConst*
_output_shapes
: *
dtype0*
valueB B ?
<dense_features/Cowork_Cor_indicator/to_sparse_input/NotEqualNotEqualinputs_cowork_corKdense_features/Cowork_Cor_indicator/to_sparse_input/ignore_value/x:output:0*
T0*'
_output_shapes
:??????????
;dense_features/Cowork_Cor_indicator/to_sparse_input/indicesWhere@dense_features/Cowork_Cor_indicator/to_sparse_input/NotEqual:z:0*'
_output_shapes
:??????????
:dense_features/Cowork_Cor_indicator/to_sparse_input/valuesGatherNdinputs_cowork_corCdense_features/Cowork_Cor_indicator/to_sparse_input/indices:index:0*
Tindices0	*
Tparams0*#
_output_shapes
:??????????
?dense_features/Cowork_Cor_indicator/to_sparse_input/dense_shapeShapeinputs_cowork_cor*
T0*
_output_shapes
:*
out_type0	?
Adense_features/Cowork_Cor_indicator/None_Lookup/LookupTableFindV2LookupTableFindV2Ndense_features_cowork_cor_indicator_none_lookup_lookuptablefindv2_table_handleCdense_features/Cowork_Cor_indicator/to_sparse_input/values:output:0Odense_features_cowork_cor_indicator_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:??????????
?dense_features/Cowork_Cor_indicator/SparseToDense/default_valueConst*
_output_shapes
: *
dtype0	*
valueB	 R
??????????
1dense_features/Cowork_Cor_indicator/SparseToDenseSparseToDenseCdense_features/Cowork_Cor_indicator/to_sparse_input/indices:index:0Hdense_features/Cowork_Cor_indicator/to_sparse_input/dense_shape:output:0Jdense_features/Cowork_Cor_indicator/None_Lookup/LookupTableFindV2:values:0Hdense_features/Cowork_Cor_indicator/SparseToDense/default_value:output:0*
T0	*
Tindices0	*'
_output_shapes
:?????????v
1dense_features/Cowork_Cor_indicator/one_hot/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??x
3dense_features/Cowork_Cor_indicator/one_hot/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    s
1dense_features/Cowork_Cor_indicator/one_hot/depthConst*
_output_shapes
: *
dtype0*
value	B :?
+dense_features/Cowork_Cor_indicator/one_hotOneHot9dense_features/Cowork_Cor_indicator/SparseToDense:dense:0:dense_features/Cowork_Cor_indicator/one_hot/depth:output:0:dense_features/Cowork_Cor_indicator/one_hot/Const:output:0<dense_features/Cowork_Cor_indicator/one_hot/Const_1:output:0*
T0*+
_output_shapes
:??????????
9dense_features/Cowork_Cor_indicator/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
??????????
'dense_features/Cowork_Cor_indicator/SumSum4dense_features/Cowork_Cor_indicator/one_hot:output:0Bdense_features/Cowork_Cor_indicator/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:??????????
)dense_features/Cowork_Cor_indicator/ShapeShape0dense_features/Cowork_Cor_indicator/Sum:output:0*
T0*
_output_shapes
:?
7dense_features/Cowork_Cor_indicator/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
9dense_features/Cowork_Cor_indicator/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
9dense_features/Cowork_Cor_indicator/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
1dense_features/Cowork_Cor_indicator/strided_sliceStridedSlice2dense_features/Cowork_Cor_indicator/Shape:output:0@dense_features/Cowork_Cor_indicator/strided_slice/stack:output:0Bdense_features/Cowork_Cor_indicator/strided_slice/stack_1:output:0Bdense_features/Cowork_Cor_indicator/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_masku
3dense_features/Cowork_Cor_indicator/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
1dense_features/Cowork_Cor_indicator/Reshape/shapePack:dense_features/Cowork_Cor_indicator/strided_slice:output:0<dense_features/Cowork_Cor_indicator/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
+dense_features/Cowork_Cor_indicator/ReshapeReshape0dense_features/Cowork_Cor_indicator/Sum:output:0:dense_features/Cowork_Cor_indicator/Reshape/shape:output:0*
T0*'
_output_shapes
:??????????
Cdense_features/Cowork_Inst_indicator/to_sparse_input/ignore_value/xConst*
_output_shapes
: *
dtype0*
valueB B ?
=dense_features/Cowork_Inst_indicator/to_sparse_input/NotEqualNotEqualinputs_cowork_instLdense_features/Cowork_Inst_indicator/to_sparse_input/ignore_value/x:output:0*
T0*'
_output_shapes
:??????????
<dense_features/Cowork_Inst_indicator/to_sparse_input/indicesWhereAdense_features/Cowork_Inst_indicator/to_sparse_input/NotEqual:z:0*'
_output_shapes
:??????????
;dense_features/Cowork_Inst_indicator/to_sparse_input/valuesGatherNdinputs_cowork_instDdense_features/Cowork_Inst_indicator/to_sparse_input/indices:index:0*
Tindices0	*
Tparams0*#
_output_shapes
:??????????
@dense_features/Cowork_Inst_indicator/to_sparse_input/dense_shapeShapeinputs_cowork_inst*
T0*
_output_shapes
:*
out_type0	?
Bdense_features/Cowork_Inst_indicator/None_Lookup/LookupTableFindV2LookupTableFindV2Odense_features_cowork_inst_indicator_none_lookup_lookuptablefindv2_table_handleDdense_features/Cowork_Inst_indicator/to_sparse_input/values:output:0Pdense_features_cowork_inst_indicator_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:??????????
@dense_features/Cowork_Inst_indicator/SparseToDense/default_valueConst*
_output_shapes
: *
dtype0	*
valueB	 R
??????????
2dense_features/Cowork_Inst_indicator/SparseToDenseSparseToDenseDdense_features/Cowork_Inst_indicator/to_sparse_input/indices:index:0Idense_features/Cowork_Inst_indicator/to_sparse_input/dense_shape:output:0Kdense_features/Cowork_Inst_indicator/None_Lookup/LookupTableFindV2:values:0Idense_features/Cowork_Inst_indicator/SparseToDense/default_value:output:0*
T0	*
Tindices0	*'
_output_shapes
:?????????w
2dense_features/Cowork_Inst_indicator/one_hot/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??y
4dense_features/Cowork_Inst_indicator/one_hot/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    t
2dense_features/Cowork_Inst_indicator/one_hot/depthConst*
_output_shapes
: *
dtype0*
value	B :?
,dense_features/Cowork_Inst_indicator/one_hotOneHot:dense_features/Cowork_Inst_indicator/SparseToDense:dense:0;dense_features/Cowork_Inst_indicator/one_hot/depth:output:0;dense_features/Cowork_Inst_indicator/one_hot/Const:output:0=dense_features/Cowork_Inst_indicator/one_hot/Const_1:output:0*
T0*+
_output_shapes
:??????????
:dense_features/Cowork_Inst_indicator/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
??????????
(dense_features/Cowork_Inst_indicator/SumSum5dense_features/Cowork_Inst_indicator/one_hot:output:0Cdense_features/Cowork_Inst_indicator/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:??????????
*dense_features/Cowork_Inst_indicator/ShapeShape1dense_features/Cowork_Inst_indicator/Sum:output:0*
T0*
_output_shapes
:?
8dense_features/Cowork_Inst_indicator/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
:dense_features/Cowork_Inst_indicator/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
:dense_features/Cowork_Inst_indicator/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
2dense_features/Cowork_Inst_indicator/strided_sliceStridedSlice3dense_features/Cowork_Inst_indicator/Shape:output:0Adense_features/Cowork_Inst_indicator/strided_slice/stack:output:0Cdense_features/Cowork_Inst_indicator/strided_slice/stack_1:output:0Cdense_features/Cowork_Inst_indicator/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskv
4dense_features/Cowork_Inst_indicator/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
2dense_features/Cowork_Inst_indicator/Reshape/shapePack;dense_features/Cowork_Inst_indicator/strided_slice:output:0=dense_features/Cowork_Inst_indicator/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
,dense_features/Cowork_Inst_indicator/ReshapeReshape1dense_features/Cowork_Inst_indicator/Sum:output:0;dense_features/Cowork_Inst_indicator/Reshape/shape:output:0*
T0*'
_output_shapes
:??????????
Bdense_features/Cowork_Uni_indicator/to_sparse_input/ignore_value/xConst*
_output_shapes
: *
dtype0*
valueB B ?
<dense_features/Cowork_Uni_indicator/to_sparse_input/NotEqualNotEqualinputs_cowork_uniKdense_features/Cowork_Uni_indicator/to_sparse_input/ignore_value/x:output:0*
T0*'
_output_shapes
:??????????
;dense_features/Cowork_Uni_indicator/to_sparse_input/indicesWhere@dense_features/Cowork_Uni_indicator/to_sparse_input/NotEqual:z:0*'
_output_shapes
:??????????
:dense_features/Cowork_Uni_indicator/to_sparse_input/valuesGatherNdinputs_cowork_uniCdense_features/Cowork_Uni_indicator/to_sparse_input/indices:index:0*
Tindices0	*
Tparams0*#
_output_shapes
:??????????
?dense_features/Cowork_Uni_indicator/to_sparse_input/dense_shapeShapeinputs_cowork_uni*
T0*
_output_shapes
:*
out_type0	?
Adense_features/Cowork_Uni_indicator/None_Lookup/LookupTableFindV2LookupTableFindV2Ndense_features_cowork_uni_indicator_none_lookup_lookuptablefindv2_table_handleCdense_features/Cowork_Uni_indicator/to_sparse_input/values:output:0Odense_features_cowork_uni_indicator_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:??????????
?dense_features/Cowork_Uni_indicator/SparseToDense/default_valueConst*
_output_shapes
: *
dtype0	*
valueB	 R
??????????
1dense_features/Cowork_Uni_indicator/SparseToDenseSparseToDenseCdense_features/Cowork_Uni_indicator/to_sparse_input/indices:index:0Hdense_features/Cowork_Uni_indicator/to_sparse_input/dense_shape:output:0Jdense_features/Cowork_Uni_indicator/None_Lookup/LookupTableFindV2:values:0Hdense_features/Cowork_Uni_indicator/SparseToDense/default_value:output:0*
T0	*
Tindices0	*'
_output_shapes
:?????????v
1dense_features/Cowork_Uni_indicator/one_hot/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??x
3dense_features/Cowork_Uni_indicator/one_hot/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    s
1dense_features/Cowork_Uni_indicator/one_hot/depthConst*
_output_shapes
: *
dtype0*
value	B :?
+dense_features/Cowork_Uni_indicator/one_hotOneHot9dense_features/Cowork_Uni_indicator/SparseToDense:dense:0:dense_features/Cowork_Uni_indicator/one_hot/depth:output:0:dense_features/Cowork_Uni_indicator/one_hot/Const:output:0<dense_features/Cowork_Uni_indicator/one_hot/Const_1:output:0*
T0*+
_output_shapes
:??????????
9dense_features/Cowork_Uni_indicator/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
??????????
'dense_features/Cowork_Uni_indicator/SumSum4dense_features/Cowork_Uni_indicator/one_hot:output:0Bdense_features/Cowork_Uni_indicator/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:??????????
)dense_features/Cowork_Uni_indicator/ShapeShape0dense_features/Cowork_Uni_indicator/Sum:output:0*
T0*
_output_shapes
:?
7dense_features/Cowork_Uni_indicator/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
9dense_features/Cowork_Uni_indicator/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
9dense_features/Cowork_Uni_indicator/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
1dense_features/Cowork_Uni_indicator/strided_sliceStridedSlice2dense_features/Cowork_Uni_indicator/Shape:output:0@dense_features/Cowork_Uni_indicator/strided_slice/stack:output:0Bdense_features/Cowork_Uni_indicator/strided_slice/stack_1:output:0Bdense_features/Cowork_Uni_indicator/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_masku
3dense_features/Cowork_Uni_indicator/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
1dense_features/Cowork_Uni_indicator/Reshape/shapePack:dense_features/Cowork_Uni_indicator/strided_slice:output:0<dense_features/Cowork_Uni_indicator/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
+dense_features/Cowork_Uni_indicator/ReshapeReshape0dense_features/Cowork_Uni_indicator/Sum:output:0:dense_features/Cowork_Uni_indicator/Reshape/shape:output:0*
T0*'
_output_shapes
:??????????
Bdense_features/Cowork_etc_indicator/to_sparse_input/ignore_value/xConst*
_output_shapes
: *
dtype0*
valueB B ?
<dense_features/Cowork_etc_indicator/to_sparse_input/NotEqualNotEqualinputs_cowork_etcKdense_features/Cowork_etc_indicator/to_sparse_input/ignore_value/x:output:0*
T0*'
_output_shapes
:??????????
;dense_features/Cowork_etc_indicator/to_sparse_input/indicesWhere@dense_features/Cowork_etc_indicator/to_sparse_input/NotEqual:z:0*'
_output_shapes
:??????????
:dense_features/Cowork_etc_indicator/to_sparse_input/valuesGatherNdinputs_cowork_etcCdense_features/Cowork_etc_indicator/to_sparse_input/indices:index:0*
Tindices0	*
Tparams0*#
_output_shapes
:??????????
?dense_features/Cowork_etc_indicator/to_sparse_input/dense_shapeShapeinputs_cowork_etc*
T0*
_output_shapes
:*
out_type0	?
Adense_features/Cowork_etc_indicator/None_Lookup/LookupTableFindV2LookupTableFindV2Ndense_features_cowork_etc_indicator_none_lookup_lookuptablefindv2_table_handleCdense_features/Cowork_etc_indicator/to_sparse_input/values:output:0Odense_features_cowork_etc_indicator_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:??????????
?dense_features/Cowork_etc_indicator/SparseToDense/default_valueConst*
_output_shapes
: *
dtype0	*
valueB	 R
??????????
1dense_features/Cowork_etc_indicator/SparseToDenseSparseToDenseCdense_features/Cowork_etc_indicator/to_sparse_input/indices:index:0Hdense_features/Cowork_etc_indicator/to_sparse_input/dense_shape:output:0Jdense_features/Cowork_etc_indicator/None_Lookup/LookupTableFindV2:values:0Hdense_features/Cowork_etc_indicator/SparseToDense/default_value:output:0*
T0	*
Tindices0	*'
_output_shapes
:?????????v
1dense_features/Cowork_etc_indicator/one_hot/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??x
3dense_features/Cowork_etc_indicator/one_hot/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    s
1dense_features/Cowork_etc_indicator/one_hot/depthConst*
_output_shapes
: *
dtype0*
value	B :?
+dense_features/Cowork_etc_indicator/one_hotOneHot9dense_features/Cowork_etc_indicator/SparseToDense:dense:0:dense_features/Cowork_etc_indicator/one_hot/depth:output:0:dense_features/Cowork_etc_indicator/one_hot/Const:output:0<dense_features/Cowork_etc_indicator/one_hot/Const_1:output:0*
T0*+
_output_shapes
:??????????
9dense_features/Cowork_etc_indicator/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
??????????
'dense_features/Cowork_etc_indicator/SumSum4dense_features/Cowork_etc_indicator/one_hot:output:0Bdense_features/Cowork_etc_indicator/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:??????????
)dense_features/Cowork_etc_indicator/ShapeShape0dense_features/Cowork_etc_indicator/Sum:output:0*
T0*
_output_shapes
:?
7dense_features/Cowork_etc_indicator/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
9dense_features/Cowork_etc_indicator/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
9dense_features/Cowork_etc_indicator/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
1dense_features/Cowork_etc_indicator/strided_sliceStridedSlice2dense_features/Cowork_etc_indicator/Shape:output:0@dense_features/Cowork_etc_indicator/strided_slice/stack:output:0Bdense_features/Cowork_etc_indicator/strided_slice/stack_1:output:0Bdense_features/Cowork_etc_indicator/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_masku
3dense_features/Cowork_etc_indicator/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
1dense_features/Cowork_etc_indicator/Reshape/shapePack:dense_features/Cowork_etc_indicator/strided_slice:output:0<dense_features/Cowork_etc_indicator/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
+dense_features/Cowork_etc_indicator/ReshapeReshape0dense_features/Cowork_etc_indicator/Sum:output:0:dense_features/Cowork_etc_indicator/Reshape/shape:output:0*
T0*'
_output_shapes
:??????????
Cdense_features/Econ_Social_indicator/to_sparse_input/ignore_value/xConst*
_output_shapes
: *
dtype0*
valueB :
??????????
Adense_features/Econ_Social_indicator/to_sparse_input/ignore_valueCastLdense_features/Econ_Social_indicator/to_sparse_input/ignore_value/x:output:0*

DstT0	*

SrcT0*
_output_shapes
: ?
=dense_features/Econ_Social_indicator/to_sparse_input/NotEqualNotEqualinputs_econ_socialEdense_features/Econ_Social_indicator/to_sparse_input/ignore_value:y:0*
T0	*'
_output_shapes
:??????????
<dense_features/Econ_Social_indicator/to_sparse_input/indicesWhereAdense_features/Econ_Social_indicator/to_sparse_input/NotEqual:z:0*'
_output_shapes
:??????????
;dense_features/Econ_Social_indicator/to_sparse_input/valuesGatherNdinputs_econ_socialDdense_features/Econ_Social_indicator/to_sparse_input/indices:index:0*
Tindices0	*
Tparams0	*#
_output_shapes
:??????????
@dense_features/Econ_Social_indicator/to_sparse_input/dense_shapeShapeinputs_econ_social*
T0	*
_output_shapes
:*
out_type0	?
Bdense_features/Econ_Social_indicator/None_Lookup/LookupTableFindV2LookupTableFindV2Odense_features_econ_social_indicator_none_lookup_lookuptablefindv2_table_handleDdense_features/Econ_Social_indicator/to_sparse_input/values:output:0Pdense_features_econ_social_indicator_none_lookup_lookuptablefindv2_default_value*	
Tin0	*

Tout0	*#
_output_shapes
:??????????
@dense_features/Econ_Social_indicator/SparseToDense/default_valueConst*
_output_shapes
: *
dtype0	*
valueB	 R
??????????
2dense_features/Econ_Social_indicator/SparseToDenseSparseToDenseDdense_features/Econ_Social_indicator/to_sparse_input/indices:index:0Idense_features/Econ_Social_indicator/to_sparse_input/dense_shape:output:0Kdense_features/Econ_Social_indicator/None_Lookup/LookupTableFindV2:values:0Idense_features/Econ_Social_indicator/SparseToDense/default_value:output:0*
T0	*
Tindices0	*'
_output_shapes
:?????????w
2dense_features/Econ_Social_indicator/one_hot/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??y
4dense_features/Econ_Social_indicator/one_hot/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    t
2dense_features/Econ_Social_indicator/one_hot/depthConst*
_output_shapes
: *
dtype0*
value	B :?
,dense_features/Econ_Social_indicator/one_hotOneHot:dense_features/Econ_Social_indicator/SparseToDense:dense:0;dense_features/Econ_Social_indicator/one_hot/depth:output:0;dense_features/Econ_Social_indicator/one_hot/Const:output:0=dense_features/Econ_Social_indicator/one_hot/Const_1:output:0*
T0*+
_output_shapes
:??????????
:dense_features/Econ_Social_indicator/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
??????????
(dense_features/Econ_Social_indicator/SumSum5dense_features/Econ_Social_indicator/one_hot:output:0Cdense_features/Econ_Social_indicator/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:??????????
*dense_features/Econ_Social_indicator/ShapeShape1dense_features/Econ_Social_indicator/Sum:output:0*
T0*
_output_shapes
:?
8dense_features/Econ_Social_indicator/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
:dense_features/Econ_Social_indicator/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
:dense_features/Econ_Social_indicator/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
2dense_features/Econ_Social_indicator/strided_sliceStridedSlice3dense_features/Econ_Social_indicator/Shape:output:0Adense_features/Econ_Social_indicator/strided_slice/stack:output:0Cdense_features/Econ_Social_indicator/strided_slice/stack_1:output:0Cdense_features/Econ_Social_indicator/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskv
4dense_features/Econ_Social_indicator/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
2dense_features/Econ_Social_indicator/Reshape/shapePack;dense_features/Econ_Social_indicator/strided_slice:output:0=dense_features/Econ_Social_indicator/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
,dense_features/Econ_Social_indicator/ReshapeReshape1dense_features/Econ_Social_indicator/Sum:output:0;dense_features/Econ_Social_indicator/Reshape/shape:output:0*
T0*'
_output_shapes
:??????????
Bdense_features/Green_Tech_indicator/to_sparse_input/ignore_value/xConst*
_output_shapes
: *
dtype0*
valueB :
??????????
@dense_features/Green_Tech_indicator/to_sparse_input/ignore_valueCastKdense_features/Green_Tech_indicator/to_sparse_input/ignore_value/x:output:0*

DstT0	*

SrcT0*
_output_shapes
: ?
<dense_features/Green_Tech_indicator/to_sparse_input/NotEqualNotEqualinputs_green_techDdense_features/Green_Tech_indicator/to_sparse_input/ignore_value:y:0*
T0	*'
_output_shapes
:??????????
;dense_features/Green_Tech_indicator/to_sparse_input/indicesWhere@dense_features/Green_Tech_indicator/to_sparse_input/NotEqual:z:0*'
_output_shapes
:??????????
:dense_features/Green_Tech_indicator/to_sparse_input/valuesGatherNdinputs_green_techCdense_features/Green_Tech_indicator/to_sparse_input/indices:index:0*
Tindices0	*
Tparams0	*#
_output_shapes
:??????????
?dense_features/Green_Tech_indicator/to_sparse_input/dense_shapeShapeinputs_green_tech*
T0	*
_output_shapes
:*
out_type0	?
Adense_features/Green_Tech_indicator/None_Lookup/LookupTableFindV2LookupTableFindV2Ndense_features_green_tech_indicator_none_lookup_lookuptablefindv2_table_handleCdense_features/Green_Tech_indicator/to_sparse_input/values:output:0Odense_features_green_tech_indicator_none_lookup_lookuptablefindv2_default_value*	
Tin0	*

Tout0	*#
_output_shapes
:??????????
?dense_features/Green_Tech_indicator/SparseToDense/default_valueConst*
_output_shapes
: *
dtype0	*
valueB	 R
??????????
1dense_features/Green_Tech_indicator/SparseToDenseSparseToDenseCdense_features/Green_Tech_indicator/to_sparse_input/indices:index:0Hdense_features/Green_Tech_indicator/to_sparse_input/dense_shape:output:0Jdense_features/Green_Tech_indicator/None_Lookup/LookupTableFindV2:values:0Hdense_features/Green_Tech_indicator/SparseToDense/default_value:output:0*
T0	*
Tindices0	*'
_output_shapes
:?????????v
1dense_features/Green_Tech_indicator/one_hot/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??x
3dense_features/Green_Tech_indicator/one_hot/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    s
1dense_features/Green_Tech_indicator/one_hot/depthConst*
_output_shapes
: *
dtype0*
value	B :<?
+dense_features/Green_Tech_indicator/one_hotOneHot9dense_features/Green_Tech_indicator/SparseToDense:dense:0:dense_features/Green_Tech_indicator/one_hot/depth:output:0:dense_features/Green_Tech_indicator/one_hot/Const:output:0<dense_features/Green_Tech_indicator/one_hot/Const_1:output:0*
T0*+
_output_shapes
:?????????<?
9dense_features/Green_Tech_indicator/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
??????????
'dense_features/Green_Tech_indicator/SumSum4dense_features/Green_Tech_indicator/one_hot:output:0Bdense_features/Green_Tech_indicator/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:?????????<?
)dense_features/Green_Tech_indicator/ShapeShape0dense_features/Green_Tech_indicator/Sum:output:0*
T0*
_output_shapes
:?
7dense_features/Green_Tech_indicator/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
9dense_features/Green_Tech_indicator/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
9dense_features/Green_Tech_indicator/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
1dense_features/Green_Tech_indicator/strided_sliceStridedSlice2dense_features/Green_Tech_indicator/Shape:output:0@dense_features/Green_Tech_indicator/strided_slice/stack:output:0Bdense_features/Green_Tech_indicator/strided_slice/stack_1:output:0Bdense_features/Green_Tech_indicator/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_masku
3dense_features/Green_Tech_indicator/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :<?
1dense_features/Green_Tech_indicator/Reshape/shapePack:dense_features/Green_Tech_indicator/strided_slice:output:0<dense_features/Green_Tech_indicator/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
+dense_features/Green_Tech_indicator/ReshapeReshape0dense_features/Green_Tech_indicator/Sum:output:0:dense_features/Green_Tech_indicator/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????<d
!dense_features/Log_Duration/ShapeShapeinputs_log_duration*
T0*
_output_shapes
:y
/dense_features/Log_Duration/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: {
1dense_features/Log_Duration/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:{
1dense_features/Log_Duration/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
)dense_features/Log_Duration/strided_sliceStridedSlice*dense_features/Log_Duration/Shape:output:08dense_features/Log_Duration/strided_slice/stack:output:0:dense_features/Log_Duration/strided_slice/stack_1:output:0:dense_features/Log_Duration/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskm
+dense_features/Log_Duration/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
)dense_features/Log_Duration/Reshape/shapePack2dense_features/Log_Duration/strided_slice:output:04dense_features/Log_Duration/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
#dense_features/Log_Duration/ReshapeReshapeinputs_log_duration2dense_features/Log_Duration/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????d
!dense_features/Log_RnD_Fund/ShapeShapeinputs_log_rnd_fund*
T0*
_output_shapes
:y
/dense_features/Log_RnD_Fund/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: {
1dense_features/Log_RnD_Fund/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:{
1dense_features/Log_RnD_Fund/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
)dense_features/Log_RnD_Fund/strided_sliceStridedSlice*dense_features/Log_RnD_Fund/Shape:output:08dense_features/Log_RnD_Fund/strided_slice/stack:output:0:dense_features/Log_RnD_Fund/strided_slice/stack_1:output:0:dense_features/Log_RnD_Fund/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskm
+dense_features/Log_RnD_Fund/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
)dense_features/Log_RnD_Fund/Reshape/shapePack2dense_features/Log_RnD_Fund/strided_slice:output:04dense_features/Log_RnD_Fund/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
#dense_features/Log_RnD_Fund/ReshapeReshapeinputs_log_rnd_fund2dense_features/Log_RnD_Fund/Reshape/shape:output:0*
T0*'
_output_shapes
:??????????
Bdense_features/Multi_Year_indicator/to_sparse_input/ignore_value/xConst*
_output_shapes
: *
dtype0*
valueB :
??????????
@dense_features/Multi_Year_indicator/to_sparse_input/ignore_valueCastKdense_features/Multi_Year_indicator/to_sparse_input/ignore_value/x:output:0*

DstT0	*

SrcT0*
_output_shapes
: ?
<dense_features/Multi_Year_indicator/to_sparse_input/NotEqualNotEqualinputs_multi_yearDdense_features/Multi_Year_indicator/to_sparse_input/ignore_value:y:0*
T0	*'
_output_shapes
:??????????
;dense_features/Multi_Year_indicator/to_sparse_input/indicesWhere@dense_features/Multi_Year_indicator/to_sparse_input/NotEqual:z:0*'
_output_shapes
:??????????
:dense_features/Multi_Year_indicator/to_sparse_input/valuesGatherNdinputs_multi_yearCdense_features/Multi_Year_indicator/to_sparse_input/indices:index:0*
Tindices0	*
Tparams0	*#
_output_shapes
:??????????
?dense_features/Multi_Year_indicator/to_sparse_input/dense_shapeShapeinputs_multi_year*
T0	*
_output_shapes
:*
out_type0	?
Adense_features/Multi_Year_indicator/None_Lookup/LookupTableFindV2LookupTableFindV2Ndense_features_multi_year_indicator_none_lookup_lookuptablefindv2_table_handleCdense_features/Multi_Year_indicator/to_sparse_input/values:output:0Odense_features_multi_year_indicator_none_lookup_lookuptablefindv2_default_value*	
Tin0	*

Tout0	*#
_output_shapes
:??????????
?dense_features/Multi_Year_indicator/SparseToDense/default_valueConst*
_output_shapes
: *
dtype0	*
valueB	 R
??????????
1dense_features/Multi_Year_indicator/SparseToDenseSparseToDenseCdense_features/Multi_Year_indicator/to_sparse_input/indices:index:0Hdense_features/Multi_Year_indicator/to_sparse_input/dense_shape:output:0Jdense_features/Multi_Year_indicator/None_Lookup/LookupTableFindV2:values:0Hdense_features/Multi_Year_indicator/SparseToDense/default_value:output:0*
T0	*
Tindices0	*'
_output_shapes
:?????????v
1dense_features/Multi_Year_indicator/one_hot/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??x
3dense_features/Multi_Year_indicator/one_hot/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    s
1dense_features/Multi_Year_indicator/one_hot/depthConst*
_output_shapes
: *
dtype0*
value	B :?
+dense_features/Multi_Year_indicator/one_hotOneHot9dense_features/Multi_Year_indicator/SparseToDense:dense:0:dense_features/Multi_Year_indicator/one_hot/depth:output:0:dense_features/Multi_Year_indicator/one_hot/Const:output:0<dense_features/Multi_Year_indicator/one_hot/Const_1:output:0*
T0*+
_output_shapes
:??????????
9dense_features/Multi_Year_indicator/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
??????????
'dense_features/Multi_Year_indicator/SumSum4dense_features/Multi_Year_indicator/one_hot:output:0Bdense_features/Multi_Year_indicator/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:??????????
)dense_features/Multi_Year_indicator/ShapeShape0dense_features/Multi_Year_indicator/Sum:output:0*
T0*
_output_shapes
:?
7dense_features/Multi_Year_indicator/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
9dense_features/Multi_Year_indicator/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
9dense_features/Multi_Year_indicator/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
1dense_features/Multi_Year_indicator/strided_sliceStridedSlice2dense_features/Multi_Year_indicator/Shape:output:0@dense_features/Multi_Year_indicator/strided_slice/stack:output:0Bdense_features/Multi_Year_indicator/strided_slice/stack_1:output:0Bdense_features/Multi_Year_indicator/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_masku
3dense_features/Multi_Year_indicator/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
1dense_features/Multi_Year_indicator/Reshape/shapePack:dense_features/Multi_Year_indicator/strided_slice:output:0<dense_features/Multi_Year_indicator/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
+dense_features/Multi_Year_indicator/ReshapeReshape0dense_features/Multi_Year_indicator/Sum:output:0:dense_features/Multi_Year_indicator/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????d
!dense_features/N_Patent_App/ShapeShapeinputs_n_patent_app*
T0*
_output_shapes
:y
/dense_features/N_Patent_App/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: {
1dense_features/N_Patent_App/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:{
1dense_features/N_Patent_App/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
)dense_features/N_Patent_App/strided_sliceStridedSlice*dense_features/N_Patent_App/Shape:output:08dense_features/N_Patent_App/strided_slice/stack:output:0:dense_features/N_Patent_App/strided_slice/stack_1:output:0:dense_features/N_Patent_App/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskm
+dense_features/N_Patent_App/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
)dense_features/N_Patent_App/Reshape/shapePack2dense_features/N_Patent_App/strided_slice:output:04dense_features/N_Patent_App/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
#dense_features/N_Patent_App/ReshapeReshapeinputs_n_patent_app2dense_features/N_Patent_App/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????d
!dense_features/N_Patent_Reg/ShapeShapeinputs_n_patent_reg*
T0*
_output_shapes
:y
/dense_features/N_Patent_Reg/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: {
1dense_features/N_Patent_Reg/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:{
1dense_features/N_Patent_Reg/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
)dense_features/N_Patent_Reg/strided_sliceStridedSlice*dense_features/N_Patent_Reg/Shape:output:08dense_features/N_Patent_Reg/strided_slice/stack:output:0:dense_features/N_Patent_Reg/strided_slice/stack_1:output:0:dense_features/N_Patent_Reg/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskm
+dense_features/N_Patent_Reg/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
)dense_features/N_Patent_Reg/Reshape/shapePack2dense_features/N_Patent_Reg/strided_slice:output:04dense_features/N_Patent_Reg/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
#dense_features/N_Patent_Reg/ReshapeReshapeinputs_n_patent_reg2dense_features/N_Patent_Reg/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????p
'dense_features/N_of_Korean_Patent/ShapeShapeinputs_n_of_korean_patent*
T0*
_output_shapes
:
5dense_features/N_of_Korean_Patent/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
7dense_features/N_of_Korean_Patent/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
7dense_features/N_of_Korean_Patent/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
/dense_features/N_of_Korean_Patent/strided_sliceStridedSlice0dense_features/N_of_Korean_Patent/Shape:output:0>dense_features/N_of_Korean_Patent/strided_slice/stack:output:0@dense_features/N_of_Korean_Patent/strided_slice/stack_1:output:0@dense_features/N_of_Korean_Patent/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_masks
1dense_features/N_of_Korean_Patent/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
/dense_features/N_of_Korean_Patent/Reshape/shapePack8dense_features/N_of_Korean_Patent/strided_slice:output:0:dense_features/N_of_Korean_Patent/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
)dense_features/N_of_Korean_Patent/ReshapeReshapeinputs_n_of_korean_patent8dense_features/N_of_Korean_Patent/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????`
dense_features/N_of_Paper/ShapeShapeinputs_n_of_paper*
T0*
_output_shapes
:w
-dense_features/N_of_Paper/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: y
/dense_features/N_of_Paper/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:y
/dense_features/N_of_Paper/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
'dense_features/N_of_Paper/strided_sliceStridedSlice(dense_features/N_of_Paper/Shape:output:06dense_features/N_of_Paper/strided_slice/stack:output:08dense_features/N_of_Paper/strided_slice/stack_1:output:08dense_features/N_of_Paper/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskk
)dense_features/N_of_Paper/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
'dense_features/N_of_Paper/Reshape/shapePack0dense_features/N_of_Paper/strided_slice:output:02dense_features/N_of_Paper/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
!dense_features/N_of_Paper/ReshapeReshapeinputs_n_of_paper0dense_features/N_of_Paper/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????\
dense_features/N_of_SCI/ShapeShapeinputs_n_of_sci*
T0*
_output_shapes
:u
+dense_features/N_of_SCI/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: w
-dense_features/N_of_SCI/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:w
-dense_features/N_of_SCI/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
%dense_features/N_of_SCI/strided_sliceStridedSlice&dense_features/N_of_SCI/Shape:output:04dense_features/N_of_SCI/strided_slice/stack:output:06dense_features/N_of_SCI/strided_slice/stack_1:output:06dense_features/N_of_SCI/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maski
'dense_features/N_of_SCI/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
%dense_features/N_of_SCI/Reshape/shapePack.dense_features/N_of_SCI/strided_slice:output:00dense_features/N_of_SCI/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
dense_features/N_of_SCI/ReshapeReshapeinputs_n_of_sci.dense_features/N_of_SCI/Reshape/shape:output:0*
T0*'
_output_shapes
:??????????
Kdense_features/National_Strategy_2_indicator/to_sparse_input/ignore_value/xConst*
_output_shapes
: *
dtype0*
valueB :
??????????
Idense_features/National_Strategy_2_indicator/to_sparse_input/ignore_valueCastTdense_features/National_Strategy_2_indicator/to_sparse_input/ignore_value/x:output:0*

DstT0	*

SrcT0*
_output_shapes
: ?
Edense_features/National_Strategy_2_indicator/to_sparse_input/NotEqualNotEqualinputs_national_strategy_2Mdense_features/National_Strategy_2_indicator/to_sparse_input/ignore_value:y:0*
T0	*'
_output_shapes
:??????????
Ddense_features/National_Strategy_2_indicator/to_sparse_input/indicesWhereIdense_features/National_Strategy_2_indicator/to_sparse_input/NotEqual:z:0*'
_output_shapes
:??????????
Cdense_features/National_Strategy_2_indicator/to_sparse_input/valuesGatherNdinputs_national_strategy_2Ldense_features/National_Strategy_2_indicator/to_sparse_input/indices:index:0*
Tindices0	*
Tparams0	*#
_output_shapes
:??????????
Hdense_features/National_Strategy_2_indicator/to_sparse_input/dense_shapeShapeinputs_national_strategy_2*
T0	*
_output_shapes
:*
out_type0	?
Jdense_features/National_Strategy_2_indicator/None_Lookup/LookupTableFindV2LookupTableFindV2Wdense_features_national_strategy_2_indicator_none_lookup_lookuptablefindv2_table_handleLdense_features/National_Strategy_2_indicator/to_sparse_input/values:output:0Xdense_features_national_strategy_2_indicator_none_lookup_lookuptablefindv2_default_value*	
Tin0	*

Tout0	*#
_output_shapes
:??????????
Hdense_features/National_Strategy_2_indicator/SparseToDense/default_valueConst*
_output_shapes
: *
dtype0	*
valueB	 R
??????????
:dense_features/National_Strategy_2_indicator/SparseToDenseSparseToDenseLdense_features/National_Strategy_2_indicator/to_sparse_input/indices:index:0Qdense_features/National_Strategy_2_indicator/to_sparse_input/dense_shape:output:0Sdense_features/National_Strategy_2_indicator/None_Lookup/LookupTableFindV2:values:0Qdense_features/National_Strategy_2_indicator/SparseToDense/default_value:output:0*
T0	*
Tindices0	*'
_output_shapes
:?????????
:dense_features/National_Strategy_2_indicator/one_hot/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
<dense_features/National_Strategy_2_indicator/one_hot/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    |
:dense_features/National_Strategy_2_indicator/one_hot/depthConst*
_output_shapes
: *
dtype0*
value	B :?
4dense_features/National_Strategy_2_indicator/one_hotOneHotBdense_features/National_Strategy_2_indicator/SparseToDense:dense:0Cdense_features/National_Strategy_2_indicator/one_hot/depth:output:0Cdense_features/National_Strategy_2_indicator/one_hot/Const:output:0Edense_features/National_Strategy_2_indicator/one_hot/Const_1:output:0*
T0*+
_output_shapes
:??????????
Bdense_features/National_Strategy_2_indicator/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
??????????
0dense_features/National_Strategy_2_indicator/SumSum=dense_features/National_Strategy_2_indicator/one_hot:output:0Kdense_features/National_Strategy_2_indicator/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:??????????
2dense_features/National_Strategy_2_indicator/ShapeShape9dense_features/National_Strategy_2_indicator/Sum:output:0*
T0*
_output_shapes
:?
@dense_features/National_Strategy_2_indicator/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
Bdense_features/National_Strategy_2_indicator/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
Bdense_features/National_Strategy_2_indicator/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
:dense_features/National_Strategy_2_indicator/strided_sliceStridedSlice;dense_features/National_Strategy_2_indicator/Shape:output:0Idense_features/National_Strategy_2_indicator/strided_slice/stack:output:0Kdense_features/National_Strategy_2_indicator/strided_slice/stack_1:output:0Kdense_features/National_Strategy_2_indicator/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask~
<dense_features/National_Strategy_2_indicator/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
:dense_features/National_Strategy_2_indicator/Reshape/shapePackCdense_features/National_Strategy_2_indicator/strided_slice:output:0Edense_features/National_Strategy_2_indicator/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
4dense_features/National_Strategy_2_indicator/ReshapeReshape9dense_features/National_Strategy_2_indicator/Sum:output:0Cdense_features/National_Strategy_2_indicator/Reshape/shape:output:0*
T0*'
_output_shapes
:??????????
?dense_features/RnD_Org_indicator/to_sparse_input/ignore_value/xConst*
_output_shapes
: *
dtype0*
valueB :
??????????
=dense_features/RnD_Org_indicator/to_sparse_input/ignore_valueCastHdense_features/RnD_Org_indicator/to_sparse_input/ignore_value/x:output:0*

DstT0	*

SrcT0*
_output_shapes
: ?
9dense_features/RnD_Org_indicator/to_sparse_input/NotEqualNotEqualinputs_rnd_orgAdense_features/RnD_Org_indicator/to_sparse_input/ignore_value:y:0*
T0	*'
_output_shapes
:??????????
8dense_features/RnD_Org_indicator/to_sparse_input/indicesWhere=dense_features/RnD_Org_indicator/to_sparse_input/NotEqual:z:0*'
_output_shapes
:??????????
7dense_features/RnD_Org_indicator/to_sparse_input/valuesGatherNdinputs_rnd_org@dense_features/RnD_Org_indicator/to_sparse_input/indices:index:0*
Tindices0	*
Tparams0	*#
_output_shapes
:??????????
<dense_features/RnD_Org_indicator/to_sparse_input/dense_shapeShapeinputs_rnd_org*
T0	*
_output_shapes
:*
out_type0	?
>dense_features/RnD_Org_indicator/None_Lookup/LookupTableFindV2LookupTableFindV2Kdense_features_rnd_org_indicator_none_lookup_lookuptablefindv2_table_handle@dense_features/RnD_Org_indicator/to_sparse_input/values:output:0Ldense_features_rnd_org_indicator_none_lookup_lookuptablefindv2_default_value*	
Tin0	*

Tout0	*#
_output_shapes
:??????????
<dense_features/RnD_Org_indicator/SparseToDense/default_valueConst*
_output_shapes
: *
dtype0	*
valueB	 R
??????????
.dense_features/RnD_Org_indicator/SparseToDenseSparseToDense@dense_features/RnD_Org_indicator/to_sparse_input/indices:index:0Edense_features/RnD_Org_indicator/to_sparse_input/dense_shape:output:0Gdense_features/RnD_Org_indicator/None_Lookup/LookupTableFindV2:values:0Edense_features/RnD_Org_indicator/SparseToDense/default_value:output:0*
T0	*
Tindices0	*'
_output_shapes
:?????????s
.dense_features/RnD_Org_indicator/one_hot/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??u
0dense_features/RnD_Org_indicator/one_hot/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    p
.dense_features/RnD_Org_indicator/one_hot/depthConst*
_output_shapes
: *
dtype0*
value	B :?
(dense_features/RnD_Org_indicator/one_hotOneHot6dense_features/RnD_Org_indicator/SparseToDense:dense:07dense_features/RnD_Org_indicator/one_hot/depth:output:07dense_features/RnD_Org_indicator/one_hot/Const:output:09dense_features/RnD_Org_indicator/one_hot/Const_1:output:0*
T0*+
_output_shapes
:??????????
6dense_features/RnD_Org_indicator/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
??????????
$dense_features/RnD_Org_indicator/SumSum1dense_features/RnD_Org_indicator/one_hot:output:0?dense_features/RnD_Org_indicator/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:??????????
&dense_features/RnD_Org_indicator/ShapeShape-dense_features/RnD_Org_indicator/Sum:output:0*
T0*
_output_shapes
:~
4dense_features/RnD_Org_indicator/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
6dense_features/RnD_Org_indicator/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
6dense_features/RnD_Org_indicator/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
.dense_features/RnD_Org_indicator/strided_sliceStridedSlice/dense_features/RnD_Org_indicator/Shape:output:0=dense_features/RnD_Org_indicator/strided_slice/stack:output:0?dense_features/RnD_Org_indicator/strided_slice/stack_1:output:0?dense_features/RnD_Org_indicator/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskr
0dense_features/RnD_Org_indicator/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
.dense_features/RnD_Org_indicator/Reshape/shapePack7dense_features/RnD_Org_indicator/strided_slice:output:09dense_features/RnD_Org_indicator/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
(dense_features/RnD_Org_indicator/ReshapeReshape-dense_features/RnD_Org_indicator/Sum:output:07dense_features/RnD_Org_indicator/Reshape/shape:output:0*
T0*'
_output_shapes
:??????????
Adense_features/RnD_Stage_indicator/to_sparse_input/ignore_value/xConst*
_output_shapes
: *
dtype0*
valueB :
??????????
?dense_features/RnD_Stage_indicator/to_sparse_input/ignore_valueCastJdense_features/RnD_Stage_indicator/to_sparse_input/ignore_value/x:output:0*

DstT0	*

SrcT0*
_output_shapes
: ?
;dense_features/RnD_Stage_indicator/to_sparse_input/NotEqualNotEqualinputs_rnd_stageCdense_features/RnD_Stage_indicator/to_sparse_input/ignore_value:y:0*
T0	*'
_output_shapes
:??????????
:dense_features/RnD_Stage_indicator/to_sparse_input/indicesWhere?dense_features/RnD_Stage_indicator/to_sparse_input/NotEqual:z:0*'
_output_shapes
:??????????
9dense_features/RnD_Stage_indicator/to_sparse_input/valuesGatherNdinputs_rnd_stageBdense_features/RnD_Stage_indicator/to_sparse_input/indices:index:0*
Tindices0	*
Tparams0	*#
_output_shapes
:??????????
>dense_features/RnD_Stage_indicator/to_sparse_input/dense_shapeShapeinputs_rnd_stage*
T0	*
_output_shapes
:*
out_type0	?
@dense_features/RnD_Stage_indicator/None_Lookup/LookupTableFindV2LookupTableFindV2Mdense_features_rnd_stage_indicator_none_lookup_lookuptablefindv2_table_handleBdense_features/RnD_Stage_indicator/to_sparse_input/values:output:0Ndense_features_rnd_stage_indicator_none_lookup_lookuptablefindv2_default_value*	
Tin0	*

Tout0	*#
_output_shapes
:??????????
>dense_features/RnD_Stage_indicator/SparseToDense/default_valueConst*
_output_shapes
: *
dtype0	*
valueB	 R
??????????
0dense_features/RnD_Stage_indicator/SparseToDenseSparseToDenseBdense_features/RnD_Stage_indicator/to_sparse_input/indices:index:0Gdense_features/RnD_Stage_indicator/to_sparse_input/dense_shape:output:0Idense_features/RnD_Stage_indicator/None_Lookup/LookupTableFindV2:values:0Gdense_features/RnD_Stage_indicator/SparseToDense/default_value:output:0*
T0	*
Tindices0	*'
_output_shapes
:?????????u
0dense_features/RnD_Stage_indicator/one_hot/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??w
2dense_features/RnD_Stage_indicator/one_hot/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    r
0dense_features/RnD_Stage_indicator/one_hot/depthConst*
_output_shapes
: *
dtype0*
value	B :?
*dense_features/RnD_Stage_indicator/one_hotOneHot8dense_features/RnD_Stage_indicator/SparseToDense:dense:09dense_features/RnD_Stage_indicator/one_hot/depth:output:09dense_features/RnD_Stage_indicator/one_hot/Const:output:0;dense_features/RnD_Stage_indicator/one_hot/Const_1:output:0*
T0*+
_output_shapes
:??????????
8dense_features/RnD_Stage_indicator/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
??????????
&dense_features/RnD_Stage_indicator/SumSum3dense_features/RnD_Stage_indicator/one_hot:output:0Adense_features/RnD_Stage_indicator/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:??????????
(dense_features/RnD_Stage_indicator/ShapeShape/dense_features/RnD_Stage_indicator/Sum:output:0*
T0*
_output_shapes
:?
6dense_features/RnD_Stage_indicator/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
8dense_features/RnD_Stage_indicator/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
8dense_features/RnD_Stage_indicator/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
0dense_features/RnD_Stage_indicator/strided_sliceStridedSlice1dense_features/RnD_Stage_indicator/Shape:output:0?dense_features/RnD_Stage_indicator/strided_slice/stack:output:0Adense_features/RnD_Stage_indicator/strided_slice/stack_1:output:0Adense_features/RnD_Stage_indicator/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskt
2dense_features/RnD_Stage_indicator/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
0dense_features/RnD_Stage_indicator/Reshape/shapePack9dense_features/RnD_Stage_indicator/strided_slice:output:0;dense_features/RnD_Stage_indicator/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
*dense_features/RnD_Stage_indicator/ReshapeReshape/dense_features/RnD_Stage_indicator/Sum:output:09dense_features/RnD_Stage_indicator/Reshape/shape:output:0*
T0*'
_output_shapes
:??????????
Cdense_features/STP_Code_11_indicator/to_sparse_input/ignore_value/xConst*
_output_shapes
: *
dtype0*
valueB B ?
=dense_features/STP_Code_11_indicator/to_sparse_input/NotEqualNotEqualinputs_stp_code_11Ldense_features/STP_Code_11_indicator/to_sparse_input/ignore_value/x:output:0*
T0*'
_output_shapes
:??????????
<dense_features/STP_Code_11_indicator/to_sparse_input/indicesWhereAdense_features/STP_Code_11_indicator/to_sparse_input/NotEqual:z:0*'
_output_shapes
:??????????
;dense_features/STP_Code_11_indicator/to_sparse_input/valuesGatherNdinputs_stp_code_11Ddense_features/STP_Code_11_indicator/to_sparse_input/indices:index:0*
Tindices0	*
Tparams0*#
_output_shapes
:??????????
@dense_features/STP_Code_11_indicator/to_sparse_input/dense_shapeShapeinputs_stp_code_11*
T0*
_output_shapes
:*
out_type0	?
Bdense_features/STP_Code_11_indicator/None_Lookup/LookupTableFindV2LookupTableFindV2Odense_features_stp_code_11_indicator_none_lookup_lookuptablefindv2_table_handleDdense_features/STP_Code_11_indicator/to_sparse_input/values:output:0Pdense_features_stp_code_11_indicator_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:??????????
@dense_features/STP_Code_11_indicator/SparseToDense/default_valueConst*
_output_shapes
: *
dtype0	*
valueB	 R
??????????
2dense_features/STP_Code_11_indicator/SparseToDenseSparseToDenseDdense_features/STP_Code_11_indicator/to_sparse_input/indices:index:0Idense_features/STP_Code_11_indicator/to_sparse_input/dense_shape:output:0Kdense_features/STP_Code_11_indicator/None_Lookup/LookupTableFindV2:values:0Idense_features/STP_Code_11_indicator/SparseToDense/default_value:output:0*
T0	*
Tindices0	*'
_output_shapes
:?????????w
2dense_features/STP_Code_11_indicator/one_hot/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??y
4dense_features/STP_Code_11_indicator/one_hot/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    u
2dense_features/STP_Code_11_indicator/one_hot/depthConst*
_output_shapes
: *
dtype0*
value
B :??
,dense_features/STP_Code_11_indicator/one_hotOneHot:dense_features/STP_Code_11_indicator/SparseToDense:dense:0;dense_features/STP_Code_11_indicator/one_hot/depth:output:0;dense_features/STP_Code_11_indicator/one_hot/Const:output:0=dense_features/STP_Code_11_indicator/one_hot/Const_1:output:0*
T0*,
_output_shapes
:???????????
:dense_features/STP_Code_11_indicator/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
??????????
(dense_features/STP_Code_11_indicator/SumSum5dense_features/STP_Code_11_indicator/one_hot:output:0Cdense_features/STP_Code_11_indicator/Sum/reduction_indices:output:0*
T0*(
_output_shapes
:???????????
*dense_features/STP_Code_11_indicator/ShapeShape1dense_features/STP_Code_11_indicator/Sum:output:0*
T0*
_output_shapes
:?
8dense_features/STP_Code_11_indicator/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
:dense_features/STP_Code_11_indicator/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
:dense_features/STP_Code_11_indicator/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
2dense_features/STP_Code_11_indicator/strided_sliceStridedSlice3dense_features/STP_Code_11_indicator/Shape:output:0Adense_features/STP_Code_11_indicator/strided_slice/stack:output:0Cdense_features/STP_Code_11_indicator/strided_slice/stack_1:output:0Cdense_features/STP_Code_11_indicator/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskw
4dense_features/STP_Code_11_indicator/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value
B :??
2dense_features/STP_Code_11_indicator/Reshape/shapePack;dense_features/STP_Code_11_indicator/strided_slice:output:0=dense_features/STP_Code_11_indicator/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
,dense_features/STP_Code_11_indicator/ReshapeReshape1dense_features/STP_Code_11_indicator/Sum:output:0;dense_features/STP_Code_11_indicator/Reshape/shape:output:0*
T0*(
_output_shapes
:??????????n
&dense_features/STP_Code_1_Weight/ShapeShapeinputs_stp_code_1_weight*
T0*
_output_shapes
:~
4dense_features/STP_Code_1_Weight/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
6dense_features/STP_Code_1_Weight/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
6dense_features/STP_Code_1_Weight/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
.dense_features/STP_Code_1_Weight/strided_sliceStridedSlice/dense_features/STP_Code_1_Weight/Shape:output:0=dense_features/STP_Code_1_Weight/strided_slice/stack:output:0?dense_features/STP_Code_1_Weight/strided_slice/stack_1:output:0?dense_features/STP_Code_1_Weight/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskr
0dense_features/STP_Code_1_Weight/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
.dense_features/STP_Code_1_Weight/Reshape/shapePack7dense_features/STP_Code_1_Weight/strided_slice:output:09dense_features/STP_Code_1_Weight/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
(dense_features/STP_Code_1_Weight/ReshapeReshapeinputs_stp_code_1_weight7dense_features/STP_Code_1_Weight/Reshape/shape:output:0*
T0*'
_output_shapes
:??????????
Cdense_features/STP_Code_21_indicator/to_sparse_input/ignore_value/xConst*
_output_shapes
: *
dtype0*
valueB B ?
=dense_features/STP_Code_21_indicator/to_sparse_input/NotEqualNotEqualinputs_stp_code_21Ldense_features/STP_Code_21_indicator/to_sparse_input/ignore_value/x:output:0*
T0*'
_output_shapes
:??????????
<dense_features/STP_Code_21_indicator/to_sparse_input/indicesWhereAdense_features/STP_Code_21_indicator/to_sparse_input/NotEqual:z:0*'
_output_shapes
:??????????
;dense_features/STP_Code_21_indicator/to_sparse_input/valuesGatherNdinputs_stp_code_21Ddense_features/STP_Code_21_indicator/to_sparse_input/indices:index:0*
Tindices0	*
Tparams0*#
_output_shapes
:??????????
@dense_features/STP_Code_21_indicator/to_sparse_input/dense_shapeShapeinputs_stp_code_21*
T0*
_output_shapes
:*
out_type0	?
Bdense_features/STP_Code_21_indicator/None_Lookup/LookupTableFindV2LookupTableFindV2Odense_features_stp_code_21_indicator_none_lookup_lookuptablefindv2_table_handleDdense_features/STP_Code_21_indicator/to_sparse_input/values:output:0Pdense_features_stp_code_21_indicator_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:??????????
@dense_features/STP_Code_21_indicator/SparseToDense/default_valueConst*
_output_shapes
: *
dtype0	*
valueB	 R
??????????
2dense_features/STP_Code_21_indicator/SparseToDenseSparseToDenseDdense_features/STP_Code_21_indicator/to_sparse_input/indices:index:0Idense_features/STP_Code_21_indicator/to_sparse_input/dense_shape:output:0Kdense_features/STP_Code_21_indicator/None_Lookup/LookupTableFindV2:values:0Idense_features/STP_Code_21_indicator/SparseToDense/default_value:output:0*
T0	*
Tindices0	*'
_output_shapes
:?????????w
2dense_features/STP_Code_21_indicator/one_hot/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??y
4dense_features/STP_Code_21_indicator/one_hot/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    u
2dense_features/STP_Code_21_indicator/one_hot/depthConst*
_output_shapes
: *
dtype0*
value
B :??
,dense_features/STP_Code_21_indicator/one_hotOneHot:dense_features/STP_Code_21_indicator/SparseToDense:dense:0;dense_features/STP_Code_21_indicator/one_hot/depth:output:0;dense_features/STP_Code_21_indicator/one_hot/Const:output:0=dense_features/STP_Code_21_indicator/one_hot/Const_1:output:0*
T0*,
_output_shapes
:???????????
:dense_features/STP_Code_21_indicator/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
??????????
(dense_features/STP_Code_21_indicator/SumSum5dense_features/STP_Code_21_indicator/one_hot:output:0Cdense_features/STP_Code_21_indicator/Sum/reduction_indices:output:0*
T0*(
_output_shapes
:???????????
*dense_features/STP_Code_21_indicator/ShapeShape1dense_features/STP_Code_21_indicator/Sum:output:0*
T0*
_output_shapes
:?
8dense_features/STP_Code_21_indicator/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
:dense_features/STP_Code_21_indicator/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
:dense_features/STP_Code_21_indicator/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
2dense_features/STP_Code_21_indicator/strided_sliceStridedSlice3dense_features/STP_Code_21_indicator/Shape:output:0Adense_features/STP_Code_21_indicator/strided_slice/stack:output:0Cdense_features/STP_Code_21_indicator/strided_slice/stack_1:output:0Cdense_features/STP_Code_21_indicator/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskw
4dense_features/STP_Code_21_indicator/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value
B :??
2dense_features/STP_Code_21_indicator/Reshape/shapePack;dense_features/STP_Code_21_indicator/strided_slice:output:0=dense_features/STP_Code_21_indicator/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
,dense_features/STP_Code_21_indicator/ReshapeReshape1dense_features/STP_Code_21_indicator/Sum:output:0;dense_features/STP_Code_21_indicator/Reshape/shape:output:0*
T0*(
_output_shapes
:??????????n
&dense_features/STP_Code_2_Weight/ShapeShapeinputs_stp_code_2_weight*
T0*
_output_shapes
:~
4dense_features/STP_Code_2_Weight/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
6dense_features/STP_Code_2_Weight/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
6dense_features/STP_Code_2_Weight/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
.dense_features/STP_Code_2_Weight/strided_sliceStridedSlice/dense_features/STP_Code_2_Weight/Shape:output:0=dense_features/STP_Code_2_Weight/strided_slice/stack:output:0?dense_features/STP_Code_2_Weight/strided_slice/stack_1:output:0?dense_features/STP_Code_2_Weight/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskr
0dense_features/STP_Code_2_Weight/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
.dense_features/STP_Code_2_Weight/Reshape/shapePack7dense_features/STP_Code_2_Weight/strided_slice:output:09dense_features/STP_Code_2_Weight/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
(dense_features/STP_Code_2_Weight/ReshapeReshapeinputs_stp_code_2_weight7dense_features/STP_Code_2_Weight/Reshape/shape:output:0*
T0*'
_output_shapes
:??????????
>dense_features/SixT_2_indicator/to_sparse_input/ignore_value/xConst*
_output_shapes
: *
dtype0*
valueB :
??????????
<dense_features/SixT_2_indicator/to_sparse_input/ignore_valueCastGdense_features/SixT_2_indicator/to_sparse_input/ignore_value/x:output:0*

DstT0	*

SrcT0*
_output_shapes
: ?
8dense_features/SixT_2_indicator/to_sparse_input/NotEqualNotEqualinputs_sixt_2@dense_features/SixT_2_indicator/to_sparse_input/ignore_value:y:0*
T0	*'
_output_shapes
:??????????
7dense_features/SixT_2_indicator/to_sparse_input/indicesWhere<dense_features/SixT_2_indicator/to_sparse_input/NotEqual:z:0*'
_output_shapes
:??????????
6dense_features/SixT_2_indicator/to_sparse_input/valuesGatherNdinputs_sixt_2?dense_features/SixT_2_indicator/to_sparse_input/indices:index:0*
Tindices0	*
Tparams0	*#
_output_shapes
:??????????
;dense_features/SixT_2_indicator/to_sparse_input/dense_shapeShapeinputs_sixt_2*
T0	*
_output_shapes
:*
out_type0	?
=dense_features/SixT_2_indicator/None_Lookup/LookupTableFindV2LookupTableFindV2Jdense_features_sixt_2_indicator_none_lookup_lookuptablefindv2_table_handle?dense_features/SixT_2_indicator/to_sparse_input/values:output:0Kdense_features_sixt_2_indicator_none_lookup_lookuptablefindv2_default_value*	
Tin0	*

Tout0	*#
_output_shapes
:??????????
;dense_features/SixT_2_indicator/SparseToDense/default_valueConst*
_output_shapes
: *
dtype0	*
valueB	 R
??????????
-dense_features/SixT_2_indicator/SparseToDenseSparseToDense?dense_features/SixT_2_indicator/to_sparse_input/indices:index:0Ddense_features/SixT_2_indicator/to_sparse_input/dense_shape:output:0Fdense_features/SixT_2_indicator/None_Lookup/LookupTableFindV2:values:0Ddense_features/SixT_2_indicator/SparseToDense/default_value:output:0*
T0	*
Tindices0	*'
_output_shapes
:?????????r
-dense_features/SixT_2_indicator/one_hot/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??t
/dense_features/SixT_2_indicator/one_hot/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    o
-dense_features/SixT_2_indicator/one_hot/depthConst*
_output_shapes
: *
dtype0*
value	B :?
'dense_features/SixT_2_indicator/one_hotOneHot5dense_features/SixT_2_indicator/SparseToDense:dense:06dense_features/SixT_2_indicator/one_hot/depth:output:06dense_features/SixT_2_indicator/one_hot/Const:output:08dense_features/SixT_2_indicator/one_hot/Const_1:output:0*
T0*+
_output_shapes
:??????????
5dense_features/SixT_2_indicator/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
??????????
#dense_features/SixT_2_indicator/SumSum0dense_features/SixT_2_indicator/one_hot:output:0>dense_features/SixT_2_indicator/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:??????????
%dense_features/SixT_2_indicator/ShapeShape,dense_features/SixT_2_indicator/Sum:output:0*
T0*
_output_shapes
:}
3dense_features/SixT_2_indicator/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
5dense_features/SixT_2_indicator/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
5dense_features/SixT_2_indicator/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
-dense_features/SixT_2_indicator/strided_sliceStridedSlice.dense_features/SixT_2_indicator/Shape:output:0<dense_features/SixT_2_indicator/strided_slice/stack:output:0>dense_features/SixT_2_indicator/strided_slice/stack_1:output:0>dense_features/SixT_2_indicator/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskq
/dense_features/SixT_2_indicator/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
-dense_features/SixT_2_indicator/Reshape/shapePack6dense_features/SixT_2_indicator/strided_slice:output:08dense_features/SixT_2_indicator/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
'dense_features/SixT_2_indicator/ReshapeReshape,dense_features/SixT_2_indicator/Sum:output:06dense_features/SixT_2_indicator/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????T
dense_features/Year/ShapeShapeinputs_year*
T0*
_output_shapes
:q
'dense_features/Year/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: s
)dense_features/Year/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:s
)dense_features/Year/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
!dense_features/Year/strided_sliceStridedSlice"dense_features/Year/Shape:output:00dense_features/Year/strided_slice/stack:output:02dense_features/Year/strided_slice/stack_1:output:02dense_features/Year/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maske
#dense_features/Year/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
!dense_features/Year/Reshape/shapePack*dense_features/Year/strided_slice:output:0,dense_features/Year/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
dense_features/Year/ReshapeReshapeinputs_year*dense_features/Year/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????e
dense_features/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
??????????
dense_features/concatConcatV29dense_features/Application_Area_1_Weight/Reshape:output:0<dense_features/Application_Area_1_indicator/Reshape:output:09dense_features/Application_Area_2_Weight/Reshape:output:0<dense_features/Application_Area_2_indicator/Reshape:output:07dense_features/Cowork_Abroad_indicator/Reshape:output:04dense_features/Cowork_Cor_indicator/Reshape:output:05dense_features/Cowork_Inst_indicator/Reshape:output:04dense_features/Cowork_Uni_indicator/Reshape:output:04dense_features/Cowork_etc_indicator/Reshape:output:05dense_features/Econ_Social_indicator/Reshape:output:04dense_features/Green_Tech_indicator/Reshape:output:0,dense_features/Log_Duration/Reshape:output:0,dense_features/Log_RnD_Fund/Reshape:output:04dense_features/Multi_Year_indicator/Reshape:output:0,dense_features/N_Patent_App/Reshape:output:0,dense_features/N_Patent_Reg/Reshape:output:02dense_features/N_of_Korean_Patent/Reshape:output:0*dense_features/N_of_Paper/Reshape:output:0(dense_features/N_of_SCI/Reshape:output:0=dense_features/National_Strategy_2_indicator/Reshape:output:01dense_features/RnD_Org_indicator/Reshape:output:03dense_features/RnD_Stage_indicator/Reshape:output:05dense_features/STP_Code_11_indicator/Reshape:output:01dense_features/STP_Code_1_Weight/Reshape:output:05dense_features/STP_Code_21_indicator/Reshape:output:01dense_features/STP_Code_2_Weight/Reshape:output:00dense_features/SixT_2_indicator/Reshape:output:0$dense_features/Year/Reshape:output:0#dense_features/concat/axis:output:0*
N*
T0*(
_output_shapes
:???????????
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype0?
dense/MatMulMatMuldense_features/concat:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????]

dense/ReluReludense/BiasAdd:output:0*
T0*(
_output_shapes
:??????????i
dropout/IdentityIdentitydense/Relu:activations:0*
T0*(
_output_shapes
:???????????
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype0?
dense_1/MatMulMatMuldropout/Identity:output:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????a
dense_1/ReluReludense_1/BiasAdd:output:0*
T0*(
_output_shapes
:??????????m
dropout_1/IdentityIdentitydense_1/Relu:activations:0*
T0*(
_output_shapes
:???????????
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype0?
dense_2/MatMulMatMuldropout_1/Identity:output:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????a
dense_2/ReluReludense_2/BiasAdd:output:0*
T0*(
_output_shapes
:??????????m
dropout_2/IdentityIdentitydense_2/Relu:activations:0*
T0*(
_output_shapes
:???????????
dense_3/MatMul/ReadVariableOpReadVariableOp&dense_3_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype0?
dense_3/MatMulMatMuldropout_2/Identity:output:0%dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
dense_3/BiasAdd/ReadVariableOpReadVariableOp'dense_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
dense_3/BiasAddBiasAdddense_3/MatMul:product:0&dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????f
dense_3/SigmoidSigmoiddense_3/BiasAdd:output:0*
T0*'
_output_shapes
:?????????b
IdentityIdentitydense_3/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp^dense_3/BiasAdd/ReadVariableOp^dense_3/MatMul/ReadVariableOpJ^dense_features/Application_Area_1_indicator/None_Lookup/LookupTableFindV2J^dense_features/Application_Area_2_indicator/None_Lookup/LookupTableFindV2E^dense_features/Cowork_Abroad_indicator/None_Lookup/LookupTableFindV2B^dense_features/Cowork_Cor_indicator/None_Lookup/LookupTableFindV2C^dense_features/Cowork_Inst_indicator/None_Lookup/LookupTableFindV2B^dense_features/Cowork_Uni_indicator/None_Lookup/LookupTableFindV2B^dense_features/Cowork_etc_indicator/None_Lookup/LookupTableFindV2C^dense_features/Econ_Social_indicator/None_Lookup/LookupTableFindV2B^dense_features/Green_Tech_indicator/None_Lookup/LookupTableFindV2B^dense_features/Multi_Year_indicator/None_Lookup/LookupTableFindV2K^dense_features/National_Strategy_2_indicator/None_Lookup/LookupTableFindV2?^dense_features/RnD_Org_indicator/None_Lookup/LookupTableFindV2A^dense_features/RnD_Stage_indicator/None_Lookup/LookupTableFindV2C^dense_features/STP_Code_11_indicator/None_Lookup/LookupTableFindV2C^dense_features/STP_Code_21_indicator/None_Lookup/LookupTableFindV2>^dense_features/SixT_2_indicator/None_Lookup/LookupTableFindV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2>
dense_2/MatMul/ReadVariableOpdense_2/MatMul/ReadVariableOp2@
dense_3/BiasAdd/ReadVariableOpdense_3/BiasAdd/ReadVariableOp2>
dense_3/MatMul/ReadVariableOpdense_3/MatMul/ReadVariableOp2?
Idense_features/Application_Area_1_indicator/None_Lookup/LookupTableFindV2Idense_features/Application_Area_1_indicator/None_Lookup/LookupTableFindV22?
Idense_features/Application_Area_2_indicator/None_Lookup/LookupTableFindV2Idense_features/Application_Area_2_indicator/None_Lookup/LookupTableFindV22?
Ddense_features/Cowork_Abroad_indicator/None_Lookup/LookupTableFindV2Ddense_features/Cowork_Abroad_indicator/None_Lookup/LookupTableFindV22?
Adense_features/Cowork_Cor_indicator/None_Lookup/LookupTableFindV2Adense_features/Cowork_Cor_indicator/None_Lookup/LookupTableFindV22?
Bdense_features/Cowork_Inst_indicator/None_Lookup/LookupTableFindV2Bdense_features/Cowork_Inst_indicator/None_Lookup/LookupTableFindV22?
Adense_features/Cowork_Uni_indicator/None_Lookup/LookupTableFindV2Adense_features/Cowork_Uni_indicator/None_Lookup/LookupTableFindV22?
Adense_features/Cowork_etc_indicator/None_Lookup/LookupTableFindV2Adense_features/Cowork_etc_indicator/None_Lookup/LookupTableFindV22?
Bdense_features/Econ_Social_indicator/None_Lookup/LookupTableFindV2Bdense_features/Econ_Social_indicator/None_Lookup/LookupTableFindV22?
Adense_features/Green_Tech_indicator/None_Lookup/LookupTableFindV2Adense_features/Green_Tech_indicator/None_Lookup/LookupTableFindV22?
Adense_features/Multi_Year_indicator/None_Lookup/LookupTableFindV2Adense_features/Multi_Year_indicator/None_Lookup/LookupTableFindV22?
Jdense_features/National_Strategy_2_indicator/None_Lookup/LookupTableFindV2Jdense_features/National_Strategy_2_indicator/None_Lookup/LookupTableFindV22?
>dense_features/RnD_Org_indicator/None_Lookup/LookupTableFindV2>dense_features/RnD_Org_indicator/None_Lookup/LookupTableFindV22?
@dense_features/RnD_Stage_indicator/None_Lookup/LookupTableFindV2@dense_features/RnD_Stage_indicator/None_Lookup/LookupTableFindV22?
Bdense_features/STP_Code_11_indicator/None_Lookup/LookupTableFindV2Bdense_features/STP_Code_11_indicator/None_Lookup/LookupTableFindV22?
Bdense_features/STP_Code_21_indicator/None_Lookup/LookupTableFindV2Bdense_features/STP_Code_21_indicator/None_Lookup/LookupTableFindV22~
=dense_features/SixT_2_indicator/None_Lookup/LookupTableFindV2=dense_features/SixT_2_indicator/None_Lookup/LookupTableFindV2:b ^
'
_output_shapes
:?????????
3
_user_specified_nameinputs/Application_Area_1:ie
'
_output_shapes
:?????????
:
_user_specified_name" inputs/Application_Area_1_Weight:b^
'
_output_shapes
:?????????
3
_user_specified_nameinputs/Application_Area_2:ie
'
_output_shapes
:?????????
:
_user_specified_name" inputs/Application_Area_2_Weight:]Y
'
_output_shapes
:?????????
.
_user_specified_nameinputs/Cowork_Abroad:ZV
'
_output_shapes
:?????????
+
_user_specified_nameinputs/Cowork_Cor:[W
'
_output_shapes
:?????????
,
_user_specified_nameinputs/Cowork_Inst:ZV
'
_output_shapes
:?????????
+
_user_specified_nameinputs/Cowork_Uni:ZV
'
_output_shapes
:?????????
+
_user_specified_nameinputs/Cowork_etc:[	W
'
_output_shapes
:?????????
,
_user_specified_nameinputs/Econ_Social:Z
V
'
_output_shapes
:?????????
+
_user_specified_nameinputs/Green_Tech:\X
'
_output_shapes
:?????????
-
_user_specified_nameinputs/Log_Duration:\X
'
_output_shapes
:?????????
-
_user_specified_nameinputs/Log_RnD_Fund:ZV
'
_output_shapes
:?????????
+
_user_specified_nameinputs/Multi_Year:\X
'
_output_shapes
:?????????
-
_user_specified_nameinputs/N_Patent_App:\X
'
_output_shapes
:?????????
-
_user_specified_nameinputs/N_Patent_Reg:b^
'
_output_shapes
:?????????
3
_user_specified_nameinputs/N_of_Korean_Patent:ZV
'
_output_shapes
:?????????
+
_user_specified_nameinputs/N_of_Paper:XT
'
_output_shapes
:?????????
)
_user_specified_nameinputs/N_of_SCI:c_
'
_output_shapes
:?????????
4
_user_specified_nameinputs/National_Strategy_2:WS
'
_output_shapes
:?????????
(
_user_specified_nameinputs/RnD_Org:YU
'
_output_shapes
:?????????
*
_user_specified_nameinputs/RnD_Stage:[W
'
_output_shapes
:?????????
,
_user_specified_nameinputs/STP_Code_11:a]
'
_output_shapes
:?????????
2
_user_specified_nameinputs/STP_Code_1_Weight:[W
'
_output_shapes
:?????????
,
_user_specified_nameinputs/STP_Code_21:a]
'
_output_shapes
:?????????
2
_user_specified_nameinputs/STP_Code_2_Weight:VR
'
_output_shapes
:?????????
'
_user_specified_nameinputs/SixT_2:TP
'
_output_shapes
:?????????
%
_user_specified_nameinputs/Year:

_output_shapes
: :

_output_shapes
: :!

_output_shapes
: :#

_output_shapes
: :%

_output_shapes
: :'

_output_shapes
: :)

_output_shapes
: :+

_output_shapes
: :-

_output_shapes
: :/

_output_shapes
: :1

_output_shapes
: :3

_output_shapes
: :5

_output_shapes
: :7

_output_shapes
: :9

_output_shapes
: :;

_output_shapes
: 
?
?
__inference_<lambda>_1339622
.table_init375_lookuptableimportv2_table_handle*
&table_init375_lookuptableimportv2_keys,
(table_init375_lookuptableimportv2_values	
identity??!table_init375/LookupTableImportV2?
!table_init375/LookupTableImportV2LookupTableImportV2.table_init375_lookuptableimportv2_table_handle&table_init375_lookuptableimportv2_keys(table_init375_lookuptableimportv2_values*	
Tin0*

Tout0	*
_output_shapes
 J
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??L
IdentityIdentityConst:output:0^NoOp*
T0*
_output_shapes
: j
NoOpNoOp"^table_init375/LookupTableImportV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*!
_input_shapes
: ::2F
!table_init375/LookupTableImportV2!table_init375/LookupTableImportV2: 

_output_shapes
:: 

_output_shapes
:
?	
b
C__inference_dropout_layer_call_and_return_conditional_losses_129371

inputs
identity?R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??e
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:??????????C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????p
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????j
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:??????????Z
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:??????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
F
*__inference_dropout_2_layer_call_fn_133608

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_dropout_2_layer_call_and_return_conditional_losses_129172a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:??????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?	
d
E__inference_dropout_1_layer_call_and_return_conditional_losses_129338

inputs
identity?R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??e
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:??????????C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????p
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????j
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:??????????Z
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:??????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?

?
C__inference_dense_1_layer_call_and_return_conditional_losses_133556

inputs2
matmul_readvariableop_resource:
??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????Q
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:??????????b
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:??????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
;
__inference__creator_133907
identity??
hash_tablek

hash_tableHashTableV2*
_output_shapes
: *
	key_dtype0*
shared_name860*
value_dtype0	W
IdentityIdentityhash_table:table_handle:0^NoOp*
T0*
_output_shapes
: S
NoOpNoOp^hash_table*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 2

hash_table
hash_table
?	
b
C__inference_dropout_layer_call_and_return_conditional_losses_133536

inputs
identity?R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??e
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:??????????C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????p
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????j
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:??????????Z
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:??????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
c
E__inference_dropout_2_layer_call_and_return_conditional_losses_133618

inputs

identity_1O
IdentityIdentityinputs*
T0*(
_output_shapes
:??????????\

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:??????????"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
__inference_<lambda>_1339862
.table_init477_lookuptableimportv2_table_handle*
&table_init477_lookuptableimportv2_keys,
(table_init477_lookuptableimportv2_values	
identity??!table_init477/LookupTableImportV2?
!table_init477/LookupTableImportV2LookupTableImportV2.table_init477_lookuptableimportv2_table_handle&table_init477_lookuptableimportv2_keys(table_init477_lookuptableimportv2_values*	
Tin0*

Tout0	*
_output_shapes
 J
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??L
IdentityIdentityConst:output:0^NoOp*
T0*
_output_shapes
: j
NoOpNoOp"^table_init477/LookupTableImportV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*!
_input_shapes
: ::2F
!table_init477/LookupTableImportV2!table_init477/LookupTableImportV2: 

_output_shapes
:: 

_output_shapes
:
?S
?
F__inference_sequential_layer_call_and_return_conditional_losses_130357

inputs
inputs_1
inputs_2
inputs_3
inputs_4
inputs_5
inputs_6
inputs_7
inputs_8
inputs_9	
	inputs_10	
	inputs_11
	inputs_12
	inputs_13	
	inputs_14
	inputs_15
	inputs_16
	inputs_17
	inputs_18
	inputs_19	
	inputs_20	
	inputs_21	
	inputs_22
	inputs_23
	inputs_24
	inputs_25
	inputs_26	
	inputs_27
dense_features_130268
dense_features_130270	
dense_features_130272
dense_features_130274	
dense_features_130276
dense_features_130278	
dense_features_130280
dense_features_130282	
dense_features_130284
dense_features_130286	
dense_features_130288
dense_features_130290	
dense_features_130292
dense_features_130294	
dense_features_130296
dense_features_130298	
dense_features_130300
dense_features_130302	
dense_features_130304
dense_features_130306	
dense_features_130308
dense_features_130310	
dense_features_130312
dense_features_130314	
dense_features_130316
dense_features_130318	
dense_features_130320
dense_features_130322	
dense_features_130324
dense_features_130326	
dense_features_130328
dense_features_130330	 
dense_130333:
??
dense_130335:	?"
dense_1_130339:
??
dense_1_130341:	?"
dense_2_130345:
??
dense_2_130347:	?!
dense_3_130351:	?
dense_3_130353:
identity??dense/StatefulPartitionedCall?dense_1/StatefulPartitionedCall?dense_2/StatefulPartitionedCall?dense_3/StatefulPartitionedCall?&dense_features/StatefulPartitionedCall?dropout/StatefulPartitionedCall?!dropout_1/StatefulPartitionedCall?!dropout_2/StatefulPartitionedCall?
&dense_features/StatefulPartitionedCallStatefulPartitionedCallinputsinputs_1inputs_2inputs_3inputs_4inputs_5inputs_6inputs_7inputs_8inputs_9	inputs_10	inputs_11	inputs_12	inputs_13	inputs_14	inputs_15	inputs_16	inputs_17	inputs_18	inputs_19	inputs_20	inputs_21	inputs_22	inputs_23	inputs_24	inputs_25	inputs_26	inputs_27dense_features_130268dense_features_130270dense_features_130272dense_features_130274dense_features_130276dense_features_130278dense_features_130280dense_features_130282dense_features_130284dense_features_130286dense_features_130288dense_features_130290dense_features_130292dense_features_130294dense_features_130296dense_features_130298dense_features_130300dense_features_130302dense_features_130304dense_features_130306dense_features_130308dense_features_130310dense_features_130312dense_features_130314dense_features_130316dense_features_130318dense_features_130320dense_features_130322dense_features_130324dense_features_130326dense_features_130328dense_features_130330*G
Tin@
>2<																							*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_dense_features_layer_call_and_return_conditional_losses_130030?
dense/StatefulPartitionedCallStatefulPartitionedCall/dense_features/StatefulPartitionedCall:output:0dense_130333dense_130335*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_129113?
dropout/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_dropout_layer_call_and_return_conditional_losses_129371?
dense_1/StatefulPartitionedCallStatefulPartitionedCall(dropout/StatefulPartitionedCall:output:0dense_1_130339dense_1_130341*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_129137?
!dropout_1/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0 ^dropout/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_dropout_1_layer_call_and_return_conditional_losses_129338?
dense_2/StatefulPartitionedCallStatefulPartitionedCall*dropout_1/StatefulPartitionedCall:output:0dense_2_130345dense_2_130347*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_dense_2_layer_call_and_return_conditional_losses_129161?
!dropout_2/StatefulPartitionedCallStatefulPartitionedCall(dense_2/StatefulPartitionedCall:output:0"^dropout_1/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_dropout_2_layer_call_and_return_conditional_losses_129305?
dense_3/StatefulPartitionedCallStatefulPartitionedCall*dropout_2/StatefulPartitionedCall:output:0dense_3_130351dense_3_130353*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_dense_3_layer_call_and_return_conditional_losses_129185w
IdentityIdentity(dense_3/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall'^dense_features/StatefulPartitionedCall ^dropout/StatefulPartitionedCall"^dropout_1/StatefulPartitionedCall"^dropout_2/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall2P
&dense_features/StatefulPartitionedCall&dense_features/StatefulPartitionedCall2B
dropout/StatefulPartitionedCalldropout/StatefulPartitionedCall2F
!dropout_1/StatefulPartitionedCall!dropout_1/StatefulPartitionedCall2F
!dropout_2/StatefulPartitionedCall!dropout_2/StatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:O	K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:O
K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :!

_output_shapes
: :#

_output_shapes
: :%

_output_shapes
: :'

_output_shapes
: :)

_output_shapes
: :+

_output_shapes
: :-

_output_shapes
: :/

_output_shapes
: :1

_output_shapes
: :3

_output_shapes
: :5

_output_shapes
: :7

_output_shapes
: :9

_output_shapes
: :;

_output_shapes
: 
?

?
C__inference_dense_2_layer_call_and_return_conditional_losses_133603

inputs2
matmul_readvariableop_resource:
??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????Q
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:??????????b
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:??????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
__inference_<lambda>_1340102
.table_init583_lookuptableimportv2_table_handle*
&table_init583_lookuptableimportv2_keys	,
(table_init583_lookuptableimportv2_values	
identity??!table_init583/LookupTableImportV2?
!table_init583/LookupTableImportV2LookupTableImportV2.table_init583_lookuptableimportv2_table_handle&table_init583_lookuptableimportv2_keys(table_init583_lookuptableimportv2_values*	
Tin0	*

Tout0	*
_output_shapes
 J
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??L
IdentityIdentityConst:output:0^NoOp*
T0*
_output_shapes
: j
NoOpNoOp"^table_init583/LookupTableImportV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*!
_input_shapes
: :<:<2F
!table_init583/LookupTableImportV2!table_init583/LookupTableImportV2: 

_output_shapes
:<: 

_output_shapes
:<
?
?
__inference__initializer_1339152
.table_init859_lookuptableimportv2_table_handle*
&table_init859_lookuptableimportv2_keys,
(table_init859_lookuptableimportv2_values	
identity??!table_init859/LookupTableImportV2?
!table_init859/LookupTableImportV2LookupTableImportV2.table_init859_lookuptableimportv2_table_handle&table_init859_lookuptableimportv2_keys(table_init859_lookuptableimportv2_values*	
Tin0*

Tout0	*
_output_shapes
 G
ConstConst*
_output_shapes
: *
dtype0*
value	B :L
IdentityIdentityConst:output:0^NoOp*
T0*
_output_shapes
: j
NoOpNoOp"^table_init859/LookupTableImportV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*#
_input_shapes
: :?:?2F
!table_init859/LookupTableImportV2!table_init859/LookupTableImportV2:!

_output_shapes	
:?:!

_output_shapes	
:?
?

?
C__inference_dense_3_layer_call_and_return_conditional_losses_133650

inputs1
matmul_readvariableop_resource:	?-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????V
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:?????????Z
IdentityIdentitySigmoid:y:0^NoOp*
T0*'
_output_shapes
:?????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
-
__inference__destroyer_133830
identityG
ConstConst*
_output_shapes
: *
dtype0*
value	B :E
IdentityIdentityConst:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 
?
-
__inference__destroyer_133920
identityG
ConstConst*
_output_shapes
: *
dtype0*
value	B :E
IdentityIdentityConst:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 
?
?
(__inference_dense_3_layer_call_fn_133639

inputs
unknown:	?
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_dense_3_layer_call_and_return_conditional_losses_129185o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
a
(__inference_dropout_layer_call_fn_133519

inputs
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_dropout_layer_call_and_return_conditional_losses_129371p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:??????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:??????????22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
__inference__initializer_1336992
.table_init375_lookuptableimportv2_table_handle*
&table_init375_lookuptableimportv2_keys,
(table_init375_lookuptableimportv2_values	
identity??!table_init375/LookupTableImportV2?
!table_init375/LookupTableImportV2LookupTableImportV2.table_init375_lookuptableimportv2_table_handle&table_init375_lookuptableimportv2_keys(table_init375_lookuptableimportv2_values*	
Tin0*

Tout0	*
_output_shapes
 G
ConstConst*
_output_shapes
: *
dtype0*
value	B :L
IdentityIdentityConst:output:0^NoOp*
T0*
_output_shapes
: j
NoOpNoOp"^table_init375/LookupTableImportV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*!
_input_shapes
: ::2F
!table_init375/LookupTableImportV2!table_init375/LookupTableImportV2: 

_output_shapes
:: 

_output_shapes
:
?
;
__inference__creator_133691
identity??
hash_tablek

hash_tableHashTableV2*
_output_shapes
: *
	key_dtype0*
shared_name376*
value_dtype0	W
IdentityIdentityhash_table:table_handle:0^NoOp*
T0*
_output_shapes
: S
NoOpNoOp^hash_table*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 2

hash_table
hash_table
?
?
__inference__initializer_1338612
.table_init747_lookuptableimportv2_table_handle*
&table_init747_lookuptableimportv2_keys	,
(table_init747_lookuptableimportv2_values	
identity??!table_init747/LookupTableImportV2?
!table_init747/LookupTableImportV2LookupTableImportV2.table_init747_lookuptableimportv2_table_handle&table_init747_lookuptableimportv2_keys(table_init747_lookuptableimportv2_values*	
Tin0	*

Tout0	*
_output_shapes
 G
ConstConst*
_output_shapes
: *
dtype0*
value	B :L
IdentityIdentityConst:output:0^NoOp*
T0*
_output_shapes
: j
NoOpNoOp"^table_init747/LookupTableImportV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*!
_input_shapes
: ::2F
!table_init747/LookupTableImportV2!table_init747/LookupTableImportV2: 

_output_shapes
:: 

_output_shapes
:
?
?
__inference__initializer_1337712
.table_init511_lookuptableimportv2_table_handle*
&table_init511_lookuptableimportv2_keys,
(table_init511_lookuptableimportv2_values	
identity??!table_init511/LookupTableImportV2?
!table_init511/LookupTableImportV2LookupTableImportV2.table_init511_lookuptableimportv2_table_handle&table_init511_lookuptableimportv2_keys(table_init511_lookuptableimportv2_values*	
Tin0*

Tout0	*
_output_shapes
 G
ConstConst*
_output_shapes
: *
dtype0*
value	B :L
IdentityIdentityConst:output:0^NoOp*
T0*
_output_shapes
: j
NoOpNoOp"^table_init511/LookupTableImportV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*!
_input_shapes
: ::2F
!table_init511/LookupTableImportV2!table_init511/LookupTableImportV2: 

_output_shapes
:: 

_output_shapes
:
?
-
__inference__destroyer_133740
identityG
ConstConst*
_output_shapes
: *
dtype0*
value	B :E
IdentityIdentityConst:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 
?
?
__inference_<lambda>_1340262
.table_init711_lookuptableimportv2_table_handle*
&table_init711_lookuptableimportv2_keys	,
(table_init711_lookuptableimportv2_values	
identity??!table_init711/LookupTableImportV2?
!table_init711/LookupTableImportV2LookupTableImportV2.table_init711_lookuptableimportv2_table_handle&table_init711_lookuptableimportv2_keys(table_init711_lookuptableimportv2_values*	
Tin0	*

Tout0	*
_output_shapes
 J
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??L
IdentityIdentityConst:output:0^NoOp*
T0*
_output_shapes
: j
NoOpNoOp"^table_init711/LookupTableImportV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*!
_input_shapes
: ::2F
!table_init711/LookupTableImportV2!table_init711/LookupTableImportV2: 

_output_shapes
:: 

_output_shapes
:
?
?
__inference_<lambda>_1340022
.table_init547_lookuptableimportv2_table_handle*
&table_init547_lookuptableimportv2_keys	,
(table_init547_lookuptableimportv2_values	
identity??!table_init547/LookupTableImportV2?
!table_init547/LookupTableImportV2LookupTableImportV2.table_init547_lookuptableimportv2_table_handle&table_init547_lookuptableimportv2_keys(table_init547_lookuptableimportv2_values*	
Tin0	*

Tout0	*
_output_shapes
 J
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??L
IdentityIdentityConst:output:0^NoOp*
T0*
_output_shapes
: j
NoOpNoOp"^table_init547/LookupTableImportV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*!
_input_shapes
: ::2F
!table_init547/LookupTableImportV2!table_init547/LookupTableImportV2: 

_output_shapes
:: 

_output_shapes
:"?N
saver_filename:0StatefulPartitionedCall_17:0StatefulPartitionedCall_188"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
Q
Application_Area_1;
$serving_default_Application_Area_1:0?????????
_
Application_Area_1_WeightB
+serving_default_Application_Area_1_Weight:0?????????
Q
Application_Area_2;
$serving_default_Application_Area_2:0?????????
_
Application_Area_2_WeightB
+serving_default_Application_Area_2_Weight:0?????????
G
Cowork_Abroad6
serving_default_Cowork_Abroad:0?????????
A

Cowork_Cor3
serving_default_Cowork_Cor:0?????????
C
Cowork_Inst4
serving_default_Cowork_Inst:0?????????
A

Cowork_Uni3
serving_default_Cowork_Uni:0?????????
A

Cowork_etc3
serving_default_Cowork_etc:0?????????
C
Econ_Social4
serving_default_Econ_Social:0	?????????
A

Green_Tech3
serving_default_Green_Tech:0	?????????
E
Log_Duration5
serving_default_Log_Duration:0?????????
E
Log_RnD_Fund5
serving_default_Log_RnD_Fund:0?????????
A

Multi_Year3
serving_default_Multi_Year:0	?????????
E
N_Patent_App5
serving_default_N_Patent_App:0?????????
E
N_Patent_Reg5
serving_default_N_Patent_Reg:0?????????
Q
N_of_Korean_Patent;
$serving_default_N_of_Korean_Patent:0?????????
A

N_of_Paper3
serving_default_N_of_Paper:0?????????
=
N_of_SCI1
serving_default_N_of_SCI:0?????????
S
National_Strategy_2<
%serving_default_National_Strategy_2:0	?????????
;
RnD_Org0
serving_default_RnD_Org:0	?????????
?
	RnD_Stage2
serving_default_RnD_Stage:0	?????????
C
STP_Code_114
serving_default_STP_Code_11:0?????????
O
STP_Code_1_Weight:
#serving_default_STP_Code_1_Weight:0?????????
C
STP_Code_214
serving_default_STP_Code_21:0?????????
O
STP_Code_2_Weight:
#serving_default_STP_Code_2_Weight:0?????????
9
SixT_2/
serving_default_SixT_2:0	?????????
5
Year-
serving_default_Year:0??????????
output_13
StatefulPartitionedCall_16:0?????????tensorflow/serving/predict:??
?
layer-0
layer_with_weights-0
layer-1
layer-2
layer_with_weights-1
layer-3
layer-4
layer_with_weights-2
layer-5
layer-6
layer_with_weights-3
layer-7
		optimizer

_build_input_shape
	variables
trainable_variables
regularization_losses
	keras_api

signatures
?__call__
+?&call_and_return_all_conditional_losses
?_default_save_signature"
_tf_keras_sequential
?
_feature_columns

_resources
	variables
trainable_variables
regularization_losses
	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
	variables
trainable_variables
regularization_losses
	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?

 kernel
!bias
"	variables
#trainable_variables
$regularization_losses
%	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
&	variables
'trainable_variables
(regularization_losses
)	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?

*kernel
+bias
,	variables
-trainable_variables
.regularization_losses
/	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
0	variables
1trainable_variables
2regularization_losses
3	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?

4kernel
5bias
6	variables
7trainable_variables
8regularization_losses
9	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
:iter

;beta_1

<beta_2
	=decay
>learning_ratem?m? m?!m?*m?+m?4m?5m?v?v? v?!v?*v?+v?4v?5v?"
	optimizer
 "
trackable_dict_wrapper
X
0
1
 2
!3
*4
+5
46
57"
trackable_list_wrapper
X
0
1
 2
!3
*4
+5
46
57"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables

@layers
Ametrics
Blayer_regularization_losses
Clayer_metrics
	variables
trainable_variables
regularization_losses
?__call__
?_default_save_signature
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
-
?serving_default"
signature_map
 "
trackable_list_wrapper
?
DApplication_Area_1
EApplication_Area_2
FCowork_Abroad
G
Cowork_Cor
HCowork_Inst
I
Cowork_Uni
J
Cowork_etc
KEcon_Social
L
Green_Tech
M
Multi_Year
NNational_Strategy_2
ORnD_Org
P	RnD_Stage
QSTP_Code_11
RSTP_Code_21

SSixT_2"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
Tnon_trainable_variables

Ulayers
Vmetrics
Wlayer_regularization_losses
Xlayer_metrics
	variables
trainable_variables
regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
+:)
??2sequential/dense/kernel
$:"?2sequential/dense/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
Ynon_trainable_variables

Zlayers
[metrics
\layer_regularization_losses
]layer_metrics
	variables
trainable_variables
regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
^non_trainable_variables

_layers
`metrics
alayer_regularization_losses
blayer_metrics
	variables
trainable_variables
regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
-:+
??2sequential/dense_1/kernel
&:$?2sequential/dense_1/bias
.
 0
!1"
trackable_list_wrapper
.
 0
!1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
cnon_trainable_variables

dlayers
emetrics
flayer_regularization_losses
glayer_metrics
"	variables
#trainable_variables
$regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
hnon_trainable_variables

ilayers
jmetrics
klayer_regularization_losses
llayer_metrics
&	variables
'trainable_variables
(regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
-:+
??2sequential/dense_2/kernel
&:$?2sequential/dense_2/bias
.
*0
+1"
trackable_list_wrapper
.
*0
+1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
mnon_trainable_variables

nlayers
ometrics
player_regularization_losses
qlayer_metrics
,	variables
-trainable_variables
.regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
rnon_trainable_variables

slayers
tmetrics
ulayer_regularization_losses
vlayer_metrics
0	variables
1trainable_variables
2regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
,:*	?2sequential/dense_3/kernel
%:#2sequential/dense_3/bias
.
40
51"
trackable_list_wrapper
.
40
51"
trackable_list_wrapper
 "
trackable_list_wrapper
?
wnon_trainable_variables

xlayers
ymetrics
zlayer_regularization_losses
{layer_metrics
6	variables
7trainable_variables
8regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
 "
trackable_list_wrapper
X
0
1
2
3
4
5
6
7"
trackable_list_wrapper
.
|0
}1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
=
~Application_Area_1_lookup"
_generic_user_object
=
Application_Area_2_lookup"
_generic_user_object
9
?Cowork_Abroad_lookup"
_generic_user_object
6
?Cowork_Cor_lookup"
_generic_user_object
7
?Cowork_Inst_lookup"
_generic_user_object
6
?Cowork_Uni_lookup"
_generic_user_object
6
?Cowork_etc_lookup"
_generic_user_object
7
?Econ_Social_lookup"
_generic_user_object
6
?Green_Tech_lookup"
_generic_user_object
6
?Multi_Year_lookup"
_generic_user_object
?
?National_Strategy_2_lookup"
_generic_user_object
3
?RnD_Org_lookup"
_generic_user_object
5
?RnD_Stage_lookup"
_generic_user_object
7
?STP_Code_11_lookup"
_generic_user_object
7
?STP_Code_21_lookup"
_generic_user_object
2
?SixT_2_lookup"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
R

?total

?count
?	variables
?	keras_api"
_tf_keras_metric
?
?
init_shape
?true_positives
?false_positives
?false_negatives
?weights_intermediate
?	variables
?	keras_api"
_tf_keras_metric
n
?_initializer
?_create_resource
?_initialize
?_destroy_resourceR jCustom.StaticHashTable
n
?_initializer
?_create_resource
?_initialize
?_destroy_resourceR jCustom.StaticHashTable
n
?_initializer
?_create_resource
?_initialize
?_destroy_resourceR jCustom.StaticHashTable
n
?_initializer
?_create_resource
?_initialize
?_destroy_resourceR jCustom.StaticHashTable
n
?_initializer
?_create_resource
?_initialize
?_destroy_resourceR jCustom.StaticHashTable
n
?_initializer
?_create_resource
?_initialize
?_destroy_resourceR jCustom.StaticHashTable
n
?_initializer
?_create_resource
?_initialize
?_destroy_resourceR jCustom.StaticHashTable
n
?_initializer
?_create_resource
?_initialize
?_destroy_resourceR jCustom.StaticHashTable
n
?_initializer
?_create_resource
?_initialize
?_destroy_resourceR jCustom.StaticHashTable
n
?_initializer
?_create_resource
?_initialize
?_destroy_resourceR jCustom.StaticHashTable
n
?_initializer
?_create_resource
?_initialize
?_destroy_resourceR jCustom.StaticHashTable
n
?_initializer
?_create_resource
?_initialize
?_destroy_resourceR jCustom.StaticHashTable
n
?_initializer
?_create_resource
?_initialize
?_destroy_resourceR jCustom.StaticHashTable
n
?_initializer
?_create_resource
?_initialize
?_destroy_resourceR jCustom.StaticHashTable
n
?_initializer
?_create_resource
?_initialize
?_destroy_resourceR jCustom.StaticHashTable
n
?_initializer
?_create_resource
?_initialize
?_destroy_resourceR jCustom.StaticHashTable
:  (2total
:  (2count
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
 "
trackable_list_wrapper
:  (2true_positives
:  (2false_positives
:  (2false_negatives
 :  (2weights_intermediate
@
?0
?1
?2
?3"
trackable_list_wrapper
.
?	variables"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
0:.
??2Adam/sequential/dense/kernel/m
):'?2Adam/sequential/dense/bias/m
2:0
??2 Adam/sequential/dense_1/kernel/m
+:)?2Adam/sequential/dense_1/bias/m
2:0
??2 Adam/sequential/dense_2/kernel/m
+:)?2Adam/sequential/dense_2/bias/m
1:/	?2 Adam/sequential/dense_3/kernel/m
*:(2Adam/sequential/dense_3/bias/m
0:.
??2Adam/sequential/dense/kernel/v
):'?2Adam/sequential/dense/bias/v
2:0
??2 Adam/sequential/dense_1/kernel/v
+:)?2Adam/sequential/dense_1/bias/v
2:0
??2 Adam/sequential/dense_2/kernel/v
+:)?2Adam/sequential/dense_2/bias/v
1:/	?2 Adam/sequential/dense_3/kernel/v
*:(2Adam/sequential/dense_3/bias/v
?2?
+__inference_sequential_layer_call_fn_129275
+__inference_sequential_layer_call_fn_131022
+__inference_sequential_layer_call_fn_131134
+__inference_sequential_layer_call_fn_130552?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
F__inference_sequential_layer_call_and_return_conditional_losses_131685
F__inference_sequential_layer_call_and_return_conditional_losses_132257
F__inference_sequential_layer_call_and_return_conditional_losses_130671
F__inference_sequential_layer_call_and_return_conditional_losses_130790?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
!__inference__wrapped_model_128455Application_Area_1Application_Area_1_WeightApplication_Area_2Application_Area_2_WeightCowork_Abroad
Cowork_CorCowork_Inst
Cowork_Uni
Cowork_etcEcon_Social
Green_TechLog_DurationLog_RnD_Fund
Multi_YearN_Patent_AppN_Patent_RegN_of_Korean_Patent
N_of_PaperN_of_SCINational_Strategy_2RnD_Org	RnD_StageSTP_Code_11STP_Code_1_WeightSTP_Code_21STP_Code_2_WeightSixT_2Year"?
???
FullArgSpec
args? 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
/__inference_dense_features_layer_call_fn_132353
/__inference_dense_features_layer_call_fn_132449?
???
FullArgSpecE
args=?:
jself

jfeatures
jcols_to_output_tensors

jtraining
varargs
 
varkw
 
defaults?

 
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
J__inference_dense_features_layer_call_and_return_conditional_losses_132969
J__inference_dense_features_layer_call_and_return_conditional_losses_133489?
???
FullArgSpecE
args=?:
jself

jfeatures
jcols_to_output_tensors

jtraining
varargs
 
varkw
 
defaults?

 
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
&__inference_dense_layer_call_fn_133498?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
A__inference_dense_layer_call_and_return_conditional_losses_133509?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
(__inference_dropout_layer_call_fn_133514
(__inference_dropout_layer_call_fn_133519?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
C__inference_dropout_layer_call_and_return_conditional_losses_133524
C__inference_dropout_layer_call_and_return_conditional_losses_133536?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
(__inference_dense_1_layer_call_fn_133545?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
C__inference_dense_1_layer_call_and_return_conditional_losses_133556?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
*__inference_dropout_1_layer_call_fn_133561
*__inference_dropout_1_layer_call_fn_133566?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
E__inference_dropout_1_layer_call_and_return_conditional_losses_133571
E__inference_dropout_1_layer_call_and_return_conditional_losses_133583?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
(__inference_dense_2_layer_call_fn_133592?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
C__inference_dense_2_layer_call_and_return_conditional_losses_133603?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
*__inference_dropout_2_layer_call_fn_133608
*__inference_dropout_2_layer_call_fn_133613?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
E__inference_dropout_2_layer_call_and_return_conditional_losses_133618
E__inference_dropout_2_layer_call_and_return_conditional_losses_133630?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
(__inference_dense_3_layer_call_fn_133639?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
C__inference_dense_3_layer_call_and_return_conditional_losses_133650?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
$__inference_signature_wrapper_130910Application_Area_1Application_Area_1_WeightApplication_Area_2Application_Area_2_WeightCowork_Abroad
Cowork_CorCowork_Inst
Cowork_Uni
Cowork_etcEcon_Social
Green_TechLog_DurationLog_RnD_Fund
Multi_YearN_Patent_AppN_Patent_RegN_of_Korean_Patent
N_of_PaperN_of_SCINational_Strategy_2RnD_Org	RnD_StageSTP_Code_11STP_Code_1_WeightSTP_Code_21STP_Code_2_WeightSixT_2Year"?
???
FullArgSpec
args? 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
__inference__creator_133655?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?2?
__inference__initializer_133663?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?2?
__inference__destroyer_133668?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?2?
__inference__creator_133673?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?2?
__inference__initializer_133681?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?2?
__inference__destroyer_133686?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?2?
__inference__creator_133691?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?2?
__inference__initializer_133699?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?2?
__inference__destroyer_133704?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?2?
__inference__creator_133709?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?2?
__inference__initializer_133717?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?2?
__inference__destroyer_133722?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?2?
__inference__creator_133727?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?2?
__inference__initializer_133735?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?2?
__inference__destroyer_133740?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?2?
__inference__creator_133745?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?2?
__inference__initializer_133753?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?2?
__inference__destroyer_133758?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?2?
__inference__creator_133763?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?2?
__inference__initializer_133771?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?2?
__inference__destroyer_133776?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?2?
__inference__creator_133781?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?2?
__inference__initializer_133789?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?2?
__inference__destroyer_133794?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?2?
__inference__creator_133799?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?2?
__inference__initializer_133807?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?2?
__inference__destroyer_133812?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?2?
__inference__creator_133817?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?2?
__inference__initializer_133825?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?2?
__inference__destroyer_133830?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?2?
__inference__creator_133835?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?2?
__inference__initializer_133843?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?2?
__inference__destroyer_133848?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?2?
__inference__creator_133853?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?2?
__inference__initializer_133861?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?2?
__inference__destroyer_133866?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?2?
__inference__creator_133871?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?2?
__inference__initializer_133879?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?2?
__inference__destroyer_133884?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?2?
__inference__creator_133889?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?2?
__inference__initializer_133897?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?2?
__inference__destroyer_133902?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?2?
__inference__creator_133907?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?2?
__inference__initializer_133915?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?2?
__inference__destroyer_133920?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?2?
__inference__creator_133925?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?2?
__inference__initializer_133933?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?2?
__inference__destroyer_133938?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
	J
Const
J	
Const_1
J	
Const_2
J	
Const_3
J	
Const_4
J	
Const_5
J	
Const_6
J	
Const_7
J	
Const_8
J	
Const_9
J

Const_10
J

Const_11
J

Const_12
J

Const_13
J

Const_14
J

Const_15
J

Const_16
J

Const_17
J

Const_18
J

Const_19
J

Const_20
J

Const_21
J

Const_22
J

Const_23
J

Const_24
J

Const_25
J

Const_26
J

Const_27
J

Const_28
J

Const_29
J

Const_30
J

Const_31
J

Const_32
J

Const_33
J

Const_34
J

Const_35
J

Const_36
J

Const_37
J

Const_38
J

Const_39
J

Const_40
J

Const_41
J

Const_42
J

Const_43
J

Const_44
J

Const_45
J

Const_46
J

Const_477
__inference__creator_133655?

? 
? "? 7
__inference__creator_133673?

? 
? "? 7
__inference__creator_133691?

? 
? "? 7
__inference__creator_133709?

? 
? "? 7
__inference__creator_133727?

? 
? "? 7
__inference__creator_133745?

? 
? "? 7
__inference__creator_133763?

? 
? "? 7
__inference__creator_133781?

? 
? "? 7
__inference__creator_133799?

? 
? "? 7
__inference__creator_133817?

? 
? "? 7
__inference__creator_133835?

? 
? "? 7
__inference__creator_133853?

? 
? "? 7
__inference__creator_133871?

? 
? "? 7
__inference__creator_133889?

? 
? "? 7
__inference__creator_133907?

? 
? "? 7
__inference__creator_133925?

? 
? "? 9
__inference__destroyer_133668?

? 
? "? 9
__inference__destroyer_133686?

? 
? "? 9
__inference__destroyer_133704?

? 
? "? 9
__inference__destroyer_133722?

? 
? "? 9
__inference__destroyer_133740?

? 
? "? 9
__inference__destroyer_133758?

? 
? "? 9
__inference__destroyer_133776?

? 
? "? 9
__inference__destroyer_133794?

? 
? "? 9
__inference__destroyer_133812?

? 
? "? 9
__inference__destroyer_133830?

? 
? "? 9
__inference__destroyer_133848?

? 
? "? 9
__inference__destroyer_133866?

? 
? "? 9
__inference__destroyer_133884?

? 
? "? 9
__inference__destroyer_133902?

? 
? "? 9
__inference__destroyer_133920?

? 
? "? 9
__inference__destroyer_133938?

? 
? "? B
__inference__initializer_133663~???

? 
? "? B
__inference__initializer_133681???

? 
? "? C
__inference__initializer_133699 ????

? 
? "? C
__inference__initializer_133717 ????

? 
? "? C
__inference__initializer_133735 ????

? 
? "? C
__inference__initializer_133753 ????

? 
? "? C
__inference__initializer_133771 ????

? 
? "? C
__inference__initializer_133789 ????

? 
? "? C
__inference__initializer_133807 ????

? 
? "? C
__inference__initializer_133825 ????

? 
? "? C
__inference__initializer_133843 ????

? 
? "? C
__inference__initializer_133861 ????

? 
? "? C
__inference__initializer_133879 ????

? 
? "? C
__inference__initializer_133897 ????

? 
? "? C
__inference__initializer_133915 ????

? 
? "? C
__inference__initializer_133933 ????

? 
? "? ?
!__inference__wrapped_model_128455?F~?????????????????????????????? !*+45???
???
???
B
Application_Area_1,?)
Application_Area_1?????????
P
Application_Area_1_Weight3?0
Application_Area_1_Weight?????????
B
Application_Area_2,?)
Application_Area_2?????????
P
Application_Area_2_Weight3?0
Application_Area_2_Weight?????????
8
Cowork_Abroad'?$
Cowork_Abroad?????????
2

Cowork_Cor$?!

Cowork_Cor?????????
4
Cowork_Inst%?"
Cowork_Inst?????????
2

Cowork_Uni$?!

Cowork_Uni?????????
2

Cowork_etc$?!

Cowork_etc?????????
4
Econ_Social%?"
Econ_Social?????????	
2

Green_Tech$?!

Green_Tech?????????	
6
Log_Duration&?#
Log_Duration?????????
6
Log_RnD_Fund&?#
Log_RnD_Fund?????????
2

Multi_Year$?!

Multi_Year?????????	
6
N_Patent_App&?#
N_Patent_App?????????
6
N_Patent_Reg&?#
N_Patent_Reg?????????
B
N_of_Korean_Patent,?)
N_of_Korean_Patent?????????
2

N_of_Paper$?!

N_of_Paper?????????
.
N_of_SCI"?
N_of_SCI?????????
D
National_Strategy_2-?*
National_Strategy_2?????????	
,
RnD_Org!?
RnD_Org?????????	
0
	RnD_Stage#? 
	RnD_Stage?????????	
4
STP_Code_11%?"
STP_Code_11?????????
@
STP_Code_1_Weight+?(
STP_Code_1_Weight?????????
4
STP_Code_21%?"
STP_Code_21?????????
@
STP_Code_2_Weight+?(
STP_Code_2_Weight?????????
*
SixT_2 ?
SixT_2?????????	
&
Year?
Year?????????
? "3?0
.
output_1"?
output_1??????????
C__inference_dense_1_layer_call_and_return_conditional_losses_133556^ !0?-
&?#
!?
inputs??????????
? "&?#
?
0??????????
? }
(__inference_dense_1_layer_call_fn_133545Q !0?-
&?#
!?
inputs??????????
? "????????????
C__inference_dense_2_layer_call_and_return_conditional_losses_133603^*+0?-
&?#
!?
inputs??????????
? "&?#
?
0??????????
? }
(__inference_dense_2_layer_call_fn_133592Q*+0?-
&?#
!?
inputs??????????
? "????????????
C__inference_dense_3_layer_call_and_return_conditional_losses_133650]450?-
&?#
!?
inputs??????????
? "%?"
?
0?????????
? |
(__inference_dense_3_layer_call_fn_133639P450?-
&?#
!?
inputs??????????
? "???????????
J__inference_dense_features_layer_call_and_return_conditional_losses_132969?>~?????????????????????????????????
???
???
K
Application_Area_15?2
features/Application_Area_1?????????
Y
Application_Area_1_Weight<?9
"features/Application_Area_1_Weight?????????
K
Application_Area_25?2
features/Application_Area_2?????????
Y
Application_Area_2_Weight<?9
"features/Application_Area_2_Weight?????????
A
Cowork_Abroad0?-
features/Cowork_Abroad?????????
;

Cowork_Cor-?*
features/Cowork_Cor?????????
=
Cowork_Inst.?+
features/Cowork_Inst?????????
;

Cowork_Uni-?*
features/Cowork_Uni?????????
;

Cowork_etc-?*
features/Cowork_etc?????????
=
Econ_Social.?+
features/Econ_Social?????????	
;

Green_Tech-?*
features/Green_Tech?????????	
?
Log_Duration/?,
features/Log_Duration?????????
?
Log_RnD_Fund/?,
features/Log_RnD_Fund?????????
;

Multi_Year-?*
features/Multi_Year?????????	
?
N_Patent_App/?,
features/N_Patent_App?????????
?
N_Patent_Reg/?,
features/N_Patent_Reg?????????
K
N_of_Korean_Patent5?2
features/N_of_Korean_Patent?????????
;

N_of_Paper-?*
features/N_of_Paper?????????
7
N_of_SCI+?(
features/N_of_SCI?????????
M
National_Strategy_26?3
features/National_Strategy_2?????????	
5
RnD_Org*?'
features/RnD_Org?????????	
9
	RnD_Stage,?)
features/RnD_Stage?????????	
=
STP_Code_11.?+
features/STP_Code_11?????????
I
STP_Code_1_Weight4?1
features/STP_Code_1_Weight?????????
=
STP_Code_21.?+
features/STP_Code_21?????????
I
STP_Code_2_Weight4?1
features/STP_Code_2_Weight?????????
3
SixT_2)?&
features/SixT_2?????????	
/
Year'?$
features/Year?????????

 
p 
? "&?#
?
0??????????
? ?
J__inference_dense_features_layer_call_and_return_conditional_losses_133489?>~?????????????????????????????????
???
???
K
Application_Area_15?2
features/Application_Area_1?????????
Y
Application_Area_1_Weight<?9
"features/Application_Area_1_Weight?????????
K
Application_Area_25?2
features/Application_Area_2?????????
Y
Application_Area_2_Weight<?9
"features/Application_Area_2_Weight?????????
A
Cowork_Abroad0?-
features/Cowork_Abroad?????????
;

Cowork_Cor-?*
features/Cowork_Cor?????????
=
Cowork_Inst.?+
features/Cowork_Inst?????????
;

Cowork_Uni-?*
features/Cowork_Uni?????????
;

Cowork_etc-?*
features/Cowork_etc?????????
=
Econ_Social.?+
features/Econ_Social?????????	
;

Green_Tech-?*
features/Green_Tech?????????	
?
Log_Duration/?,
features/Log_Duration?????????
?
Log_RnD_Fund/?,
features/Log_RnD_Fund?????????
;

Multi_Year-?*
features/Multi_Year?????????	
?
N_Patent_App/?,
features/N_Patent_App?????????
?
N_Patent_Reg/?,
features/N_Patent_Reg?????????
K
N_of_Korean_Patent5?2
features/N_of_Korean_Patent?????????
;

N_of_Paper-?*
features/N_of_Paper?????????
7
N_of_SCI+?(
features/N_of_SCI?????????
M
National_Strategy_26?3
features/National_Strategy_2?????????	
5
RnD_Org*?'
features/RnD_Org?????????	
9
	RnD_Stage,?)
features/RnD_Stage?????????	
=
STP_Code_11.?+
features/STP_Code_11?????????
I
STP_Code_1_Weight4?1
features/STP_Code_1_Weight?????????
=
STP_Code_21.?+
features/STP_Code_21?????????
I
STP_Code_2_Weight4?1
features/STP_Code_2_Weight?????????
3
SixT_2)?&
features/SixT_2?????????	
/
Year'?$
features/Year?????????

 
p
? "&?#
?
0??????????
? ?
/__inference_dense_features_layer_call_fn_132353?>~?????????????????????????????????
???
???
K
Application_Area_15?2
features/Application_Area_1?????????
Y
Application_Area_1_Weight<?9
"features/Application_Area_1_Weight?????????
K
Application_Area_25?2
features/Application_Area_2?????????
Y
Application_Area_2_Weight<?9
"features/Application_Area_2_Weight?????????
A
Cowork_Abroad0?-
features/Cowork_Abroad?????????
;

Cowork_Cor-?*
features/Cowork_Cor?????????
=
Cowork_Inst.?+
features/Cowork_Inst?????????
;

Cowork_Uni-?*
features/Cowork_Uni?????????
;

Cowork_etc-?*
features/Cowork_etc?????????
=
Econ_Social.?+
features/Econ_Social?????????	
;

Green_Tech-?*
features/Green_Tech?????????	
?
Log_Duration/?,
features/Log_Duration?????????
?
Log_RnD_Fund/?,
features/Log_RnD_Fund?????????
;

Multi_Year-?*
features/Multi_Year?????????	
?
N_Patent_App/?,
features/N_Patent_App?????????
?
N_Patent_Reg/?,
features/N_Patent_Reg?????????
K
N_of_Korean_Patent5?2
features/N_of_Korean_Patent?????????
;

N_of_Paper-?*
features/N_of_Paper?????????
7
N_of_SCI+?(
features/N_of_SCI?????????
M
National_Strategy_26?3
features/National_Strategy_2?????????	
5
RnD_Org*?'
features/RnD_Org?????????	
9
	RnD_Stage,?)
features/RnD_Stage?????????	
=
STP_Code_11.?+
features/STP_Code_11?????????
I
STP_Code_1_Weight4?1
features/STP_Code_1_Weight?????????
=
STP_Code_21.?+
features/STP_Code_21?????????
I
STP_Code_2_Weight4?1
features/STP_Code_2_Weight?????????
3
SixT_2)?&
features/SixT_2?????????	
/
Year'?$
features/Year?????????

 
p 
? "????????????
/__inference_dense_features_layer_call_fn_132449?>~?????????????????????????????????
???
???
K
Application_Area_15?2
features/Application_Area_1?????????
Y
Application_Area_1_Weight<?9
"features/Application_Area_1_Weight?????????
K
Application_Area_25?2
features/Application_Area_2?????????
Y
Application_Area_2_Weight<?9
"features/Application_Area_2_Weight?????????
A
Cowork_Abroad0?-
features/Cowork_Abroad?????????
;

Cowork_Cor-?*
features/Cowork_Cor?????????
=
Cowork_Inst.?+
features/Cowork_Inst?????????
;

Cowork_Uni-?*
features/Cowork_Uni?????????
;

Cowork_etc-?*
features/Cowork_etc?????????
=
Econ_Social.?+
features/Econ_Social?????????	
;

Green_Tech-?*
features/Green_Tech?????????	
?
Log_Duration/?,
features/Log_Duration?????????
?
Log_RnD_Fund/?,
features/Log_RnD_Fund?????????
;

Multi_Year-?*
features/Multi_Year?????????	
?
N_Patent_App/?,
features/N_Patent_App?????????
?
N_Patent_Reg/?,
features/N_Patent_Reg?????????
K
N_of_Korean_Patent5?2
features/N_of_Korean_Patent?????????
;

N_of_Paper-?*
features/N_of_Paper?????????
7
N_of_SCI+?(
features/N_of_SCI?????????
M
National_Strategy_26?3
features/National_Strategy_2?????????	
5
RnD_Org*?'
features/RnD_Org?????????	
9
	RnD_Stage,?)
features/RnD_Stage?????????	
=
STP_Code_11.?+
features/STP_Code_11?????????
I
STP_Code_1_Weight4?1
features/STP_Code_1_Weight?????????
=
STP_Code_21.?+
features/STP_Code_21?????????
I
STP_Code_2_Weight4?1
features/STP_Code_2_Weight?????????
3
SixT_2)?&
features/SixT_2?????????	
/
Year'?$
features/Year?????????

 
p
? "????????????
A__inference_dense_layer_call_and_return_conditional_losses_133509^0?-
&?#
!?
inputs??????????
? "&?#
?
0??????????
? {
&__inference_dense_layer_call_fn_133498Q0?-
&?#
!?
inputs??????????
? "????????????
E__inference_dropout_1_layer_call_and_return_conditional_losses_133571^4?1
*?'
!?
inputs??????????
p 
? "&?#
?
0??????????
? ?
E__inference_dropout_1_layer_call_and_return_conditional_losses_133583^4?1
*?'
!?
inputs??????????
p
? "&?#
?
0??????????
? 
*__inference_dropout_1_layer_call_fn_133561Q4?1
*?'
!?
inputs??????????
p 
? "???????????
*__inference_dropout_1_layer_call_fn_133566Q4?1
*?'
!?
inputs??????????
p
? "????????????
E__inference_dropout_2_layer_call_and_return_conditional_losses_133618^4?1
*?'
!?
inputs??????????
p 
? "&?#
?
0??????????
? ?
E__inference_dropout_2_layer_call_and_return_conditional_losses_133630^4?1
*?'
!?
inputs??????????
p
? "&?#
?
0??????????
? 
*__inference_dropout_2_layer_call_fn_133608Q4?1
*?'
!?
inputs??????????
p 
? "???????????
*__inference_dropout_2_layer_call_fn_133613Q4?1
*?'
!?
inputs??????????
p
? "????????????
C__inference_dropout_layer_call_and_return_conditional_losses_133524^4?1
*?'
!?
inputs??????????
p 
? "&?#
?
0??????????
? ?
C__inference_dropout_layer_call_and_return_conditional_losses_133536^4?1
*?'
!?
inputs??????????
p
? "&?#
?
0??????????
? }
(__inference_dropout_layer_call_fn_133514Q4?1
*?'
!?
inputs??????????
p 
? "???????????}
(__inference_dropout_layer_call_fn_133519Q4?1
*?'
!?
inputs??????????
p
? "????????????
F__inference_sequential_layer_call_and_return_conditional_losses_130671?F~?????????????????????????????? !*+45???
???
???
B
Application_Area_1,?)
Application_Area_1?????????
P
Application_Area_1_Weight3?0
Application_Area_1_Weight?????????
B
Application_Area_2,?)
Application_Area_2?????????
P
Application_Area_2_Weight3?0
Application_Area_2_Weight?????????
8
Cowork_Abroad'?$
Cowork_Abroad?????????
2

Cowork_Cor$?!

Cowork_Cor?????????
4
Cowork_Inst%?"
Cowork_Inst?????????
2

Cowork_Uni$?!

Cowork_Uni?????????
2

Cowork_etc$?!

Cowork_etc?????????
4
Econ_Social%?"
Econ_Social?????????	
2

Green_Tech$?!

Green_Tech?????????	
6
Log_Duration&?#
Log_Duration?????????
6
Log_RnD_Fund&?#
Log_RnD_Fund?????????
2

Multi_Year$?!

Multi_Year?????????	
6
N_Patent_App&?#
N_Patent_App?????????
6
N_Patent_Reg&?#
N_Patent_Reg?????????
B
N_of_Korean_Patent,?)
N_of_Korean_Patent?????????
2

N_of_Paper$?!

N_of_Paper?????????
.
N_of_SCI"?
N_of_SCI?????????
D
National_Strategy_2-?*
National_Strategy_2?????????	
,
RnD_Org!?
RnD_Org?????????	
0
	RnD_Stage#? 
	RnD_Stage?????????	
4
STP_Code_11%?"
STP_Code_11?????????
@
STP_Code_1_Weight+?(
STP_Code_1_Weight?????????
4
STP_Code_21%?"
STP_Code_21?????????
@
STP_Code_2_Weight+?(
STP_Code_2_Weight?????????
*
SixT_2 ?
SixT_2?????????	
&
Year?
Year?????????
p 

 
? "%?"
?
0?????????
? ?
F__inference_sequential_layer_call_and_return_conditional_losses_130790?F~?????????????????????????????? !*+45???
???
???
B
Application_Area_1,?)
Application_Area_1?????????
P
Application_Area_1_Weight3?0
Application_Area_1_Weight?????????
B
Application_Area_2,?)
Application_Area_2?????????
P
Application_Area_2_Weight3?0
Application_Area_2_Weight?????????
8
Cowork_Abroad'?$
Cowork_Abroad?????????
2

Cowork_Cor$?!

Cowork_Cor?????????
4
Cowork_Inst%?"
Cowork_Inst?????????
2

Cowork_Uni$?!

Cowork_Uni?????????
2

Cowork_etc$?!

Cowork_etc?????????
4
Econ_Social%?"
Econ_Social?????????	
2

Green_Tech$?!

Green_Tech?????????	
6
Log_Duration&?#
Log_Duration?????????
6
Log_RnD_Fund&?#
Log_RnD_Fund?????????
2

Multi_Year$?!

Multi_Year?????????	
6
N_Patent_App&?#
N_Patent_App?????????
6
N_Patent_Reg&?#
N_Patent_Reg?????????
B
N_of_Korean_Patent,?)
N_of_Korean_Patent?????????
2

N_of_Paper$?!

N_of_Paper?????????
.
N_of_SCI"?
N_of_SCI?????????
D
National_Strategy_2-?*
National_Strategy_2?????????	
,
RnD_Org!?
RnD_Org?????????	
0
	RnD_Stage#? 
	RnD_Stage?????????	
4
STP_Code_11%?"
STP_Code_11?????????
@
STP_Code_1_Weight+?(
STP_Code_1_Weight?????????
4
STP_Code_21%?"
STP_Code_21?????????
@
STP_Code_2_Weight+?(
STP_Code_2_Weight?????????
*
SixT_2 ?
SixT_2?????????	
&
Year?
Year?????????
p

 
? "%?"
?
0?????????
? ?
F__inference_sequential_layer_call_and_return_conditional_losses_131685?F~?????????????????????????????? !*+45???
???
???
I
Application_Area_13?0
inputs/Application_Area_1?????????
W
Application_Area_1_Weight:?7
 inputs/Application_Area_1_Weight?????????
I
Application_Area_23?0
inputs/Application_Area_2?????????
W
Application_Area_2_Weight:?7
 inputs/Application_Area_2_Weight?????????
?
Cowork_Abroad.?+
inputs/Cowork_Abroad?????????
9

Cowork_Cor+?(
inputs/Cowork_Cor?????????
;
Cowork_Inst,?)
inputs/Cowork_Inst?????????
9

Cowork_Uni+?(
inputs/Cowork_Uni?????????
9

Cowork_etc+?(
inputs/Cowork_etc?????????
;
Econ_Social,?)
inputs/Econ_Social?????????	
9

Green_Tech+?(
inputs/Green_Tech?????????	
=
Log_Duration-?*
inputs/Log_Duration?????????
=
Log_RnD_Fund-?*
inputs/Log_RnD_Fund?????????
9

Multi_Year+?(
inputs/Multi_Year?????????	
=
N_Patent_App-?*
inputs/N_Patent_App?????????
=
N_Patent_Reg-?*
inputs/N_Patent_Reg?????????
I
N_of_Korean_Patent3?0
inputs/N_of_Korean_Patent?????????
9

N_of_Paper+?(
inputs/N_of_Paper?????????
5
N_of_SCI)?&
inputs/N_of_SCI?????????
K
National_Strategy_24?1
inputs/National_Strategy_2?????????	
3
RnD_Org(?%
inputs/RnD_Org?????????	
7
	RnD_Stage*?'
inputs/RnD_Stage?????????	
;
STP_Code_11,?)
inputs/STP_Code_11?????????
G
STP_Code_1_Weight2?/
inputs/STP_Code_1_Weight?????????
;
STP_Code_21,?)
inputs/STP_Code_21?????????
G
STP_Code_2_Weight2?/
inputs/STP_Code_2_Weight?????????
1
SixT_2'?$
inputs/SixT_2?????????	
-
Year%?"
inputs/Year?????????
p 

 
? "%?"
?
0?????????
? ?
F__inference_sequential_layer_call_and_return_conditional_losses_132257?F~?????????????????????????????? !*+45???
???
???
I
Application_Area_13?0
inputs/Application_Area_1?????????
W
Application_Area_1_Weight:?7
 inputs/Application_Area_1_Weight?????????
I
Application_Area_23?0
inputs/Application_Area_2?????????
W
Application_Area_2_Weight:?7
 inputs/Application_Area_2_Weight?????????
?
Cowork_Abroad.?+
inputs/Cowork_Abroad?????????
9

Cowork_Cor+?(
inputs/Cowork_Cor?????????
;
Cowork_Inst,?)
inputs/Cowork_Inst?????????
9

Cowork_Uni+?(
inputs/Cowork_Uni?????????
9

Cowork_etc+?(
inputs/Cowork_etc?????????
;
Econ_Social,?)
inputs/Econ_Social?????????	
9

Green_Tech+?(
inputs/Green_Tech?????????	
=
Log_Duration-?*
inputs/Log_Duration?????????
=
Log_RnD_Fund-?*
inputs/Log_RnD_Fund?????????
9

Multi_Year+?(
inputs/Multi_Year?????????	
=
N_Patent_App-?*
inputs/N_Patent_App?????????
=
N_Patent_Reg-?*
inputs/N_Patent_Reg?????????
I
N_of_Korean_Patent3?0
inputs/N_of_Korean_Patent?????????
9

N_of_Paper+?(
inputs/N_of_Paper?????????
5
N_of_SCI)?&
inputs/N_of_SCI?????????
K
National_Strategy_24?1
inputs/National_Strategy_2?????????	
3
RnD_Org(?%
inputs/RnD_Org?????????	
7
	RnD_Stage*?'
inputs/RnD_Stage?????????	
;
STP_Code_11,?)
inputs/STP_Code_11?????????
G
STP_Code_1_Weight2?/
inputs/STP_Code_1_Weight?????????
;
STP_Code_21,?)
inputs/STP_Code_21?????????
G
STP_Code_2_Weight2?/
inputs/STP_Code_2_Weight?????????
1
SixT_2'?$
inputs/SixT_2?????????	
-
Year%?"
inputs/Year?????????
p

 
? "%?"
?
0?????????
? ?
+__inference_sequential_layer_call_fn_129275?F~?????????????????????????????? !*+45???
???
???
B
Application_Area_1,?)
Application_Area_1?????????
P
Application_Area_1_Weight3?0
Application_Area_1_Weight?????????
B
Application_Area_2,?)
Application_Area_2?????????
P
Application_Area_2_Weight3?0
Application_Area_2_Weight?????????
8
Cowork_Abroad'?$
Cowork_Abroad?????????
2

Cowork_Cor$?!

Cowork_Cor?????????
4
Cowork_Inst%?"
Cowork_Inst?????????
2

Cowork_Uni$?!

Cowork_Uni?????????
2

Cowork_etc$?!

Cowork_etc?????????
4
Econ_Social%?"
Econ_Social?????????	
2

Green_Tech$?!

Green_Tech?????????	
6
Log_Duration&?#
Log_Duration?????????
6
Log_RnD_Fund&?#
Log_RnD_Fund?????????
2

Multi_Year$?!

Multi_Year?????????	
6
N_Patent_App&?#
N_Patent_App?????????
6
N_Patent_Reg&?#
N_Patent_Reg?????????
B
N_of_Korean_Patent,?)
N_of_Korean_Patent?????????
2

N_of_Paper$?!

N_of_Paper?????????
.
N_of_SCI"?
N_of_SCI?????????
D
National_Strategy_2-?*
National_Strategy_2?????????	
,
RnD_Org!?
RnD_Org?????????	
0
	RnD_Stage#? 
	RnD_Stage?????????	
4
STP_Code_11%?"
STP_Code_11?????????
@
STP_Code_1_Weight+?(
STP_Code_1_Weight?????????
4
STP_Code_21%?"
STP_Code_21?????????
@
STP_Code_2_Weight+?(
STP_Code_2_Weight?????????
*
SixT_2 ?
SixT_2?????????	
&
Year?
Year?????????
p 

 
? "???????????
+__inference_sequential_layer_call_fn_130552?F~?????????????????????????????? !*+45???
???
???
B
Application_Area_1,?)
Application_Area_1?????????
P
Application_Area_1_Weight3?0
Application_Area_1_Weight?????????
B
Application_Area_2,?)
Application_Area_2?????????
P
Application_Area_2_Weight3?0
Application_Area_2_Weight?????????
8
Cowork_Abroad'?$
Cowork_Abroad?????????
2

Cowork_Cor$?!

Cowork_Cor?????????
4
Cowork_Inst%?"
Cowork_Inst?????????
2

Cowork_Uni$?!

Cowork_Uni?????????
2

Cowork_etc$?!

Cowork_etc?????????
4
Econ_Social%?"
Econ_Social?????????	
2

Green_Tech$?!

Green_Tech?????????	
6
Log_Duration&?#
Log_Duration?????????
6
Log_RnD_Fund&?#
Log_RnD_Fund?????????
2

Multi_Year$?!

Multi_Year?????????	
6
N_Patent_App&?#
N_Patent_App?????????
6
N_Patent_Reg&?#
N_Patent_Reg?????????
B
N_of_Korean_Patent,?)
N_of_Korean_Patent?????????
2

N_of_Paper$?!

N_of_Paper?????????
.
N_of_SCI"?
N_of_SCI?????????
D
National_Strategy_2-?*
National_Strategy_2?????????	
,
RnD_Org!?
RnD_Org?????????	
0
	RnD_Stage#? 
	RnD_Stage?????????	
4
STP_Code_11%?"
STP_Code_11?????????
@
STP_Code_1_Weight+?(
STP_Code_1_Weight?????????
4
STP_Code_21%?"
STP_Code_21?????????
@
STP_Code_2_Weight+?(
STP_Code_2_Weight?????????
*
SixT_2 ?
SixT_2?????????	
&
Year?
Year?????????
p

 
? "???????????
+__inference_sequential_layer_call_fn_131022?F~?????????????????????????????? !*+45???
???
???
I
Application_Area_13?0
inputs/Application_Area_1?????????
W
Application_Area_1_Weight:?7
 inputs/Application_Area_1_Weight?????????
I
Application_Area_23?0
inputs/Application_Area_2?????????
W
Application_Area_2_Weight:?7
 inputs/Application_Area_2_Weight?????????
?
Cowork_Abroad.?+
inputs/Cowork_Abroad?????????
9

Cowork_Cor+?(
inputs/Cowork_Cor?????????
;
Cowork_Inst,?)
inputs/Cowork_Inst?????????
9

Cowork_Uni+?(
inputs/Cowork_Uni?????????
9

Cowork_etc+?(
inputs/Cowork_etc?????????
;
Econ_Social,?)
inputs/Econ_Social?????????	
9

Green_Tech+?(
inputs/Green_Tech?????????	
=
Log_Duration-?*
inputs/Log_Duration?????????
=
Log_RnD_Fund-?*
inputs/Log_RnD_Fund?????????
9

Multi_Year+?(
inputs/Multi_Year?????????	
=
N_Patent_App-?*
inputs/N_Patent_App?????????
=
N_Patent_Reg-?*
inputs/N_Patent_Reg?????????
I
N_of_Korean_Patent3?0
inputs/N_of_Korean_Patent?????????
9

N_of_Paper+?(
inputs/N_of_Paper?????????
5
N_of_SCI)?&
inputs/N_of_SCI?????????
K
National_Strategy_24?1
inputs/National_Strategy_2?????????	
3
RnD_Org(?%
inputs/RnD_Org?????????	
7
	RnD_Stage*?'
inputs/RnD_Stage?????????	
;
STP_Code_11,?)
inputs/STP_Code_11?????????
G
STP_Code_1_Weight2?/
inputs/STP_Code_1_Weight?????????
;
STP_Code_21,?)
inputs/STP_Code_21?????????
G
STP_Code_2_Weight2?/
inputs/STP_Code_2_Weight?????????
1
SixT_2'?$
inputs/SixT_2?????????	
-
Year%?"
inputs/Year?????????
p 

 
? "???????????
+__inference_sequential_layer_call_fn_131134?F~?????????????????????????????? !*+45???
???
???
I
Application_Area_13?0
inputs/Application_Area_1?????????
W
Application_Area_1_Weight:?7
 inputs/Application_Area_1_Weight?????????
I
Application_Area_23?0
inputs/Application_Area_2?????????
W
Application_Area_2_Weight:?7
 inputs/Application_Area_2_Weight?????????
?
Cowork_Abroad.?+
inputs/Cowork_Abroad?????????
9

Cowork_Cor+?(
inputs/Cowork_Cor?????????
;
Cowork_Inst,?)
inputs/Cowork_Inst?????????
9

Cowork_Uni+?(
inputs/Cowork_Uni?????????
9

Cowork_etc+?(
inputs/Cowork_etc?????????
;
Econ_Social,?)
inputs/Econ_Social?????????	
9

Green_Tech+?(
inputs/Green_Tech?????????	
=
Log_Duration-?*
inputs/Log_Duration?????????
=
Log_RnD_Fund-?*
inputs/Log_RnD_Fund?????????
9

Multi_Year+?(
inputs/Multi_Year?????????	
=
N_Patent_App-?*
inputs/N_Patent_App?????????
=
N_Patent_Reg-?*
inputs/N_Patent_Reg?????????
I
N_of_Korean_Patent3?0
inputs/N_of_Korean_Patent?????????
9

N_of_Paper+?(
inputs/N_of_Paper?????????
5
N_of_SCI)?&
inputs/N_of_SCI?????????
K
National_Strategy_24?1
inputs/National_Strategy_2?????????	
3
RnD_Org(?%
inputs/RnD_Org?????????	
7
	RnD_Stage*?'
inputs/RnD_Stage?????????	
;
STP_Code_11,?)
inputs/STP_Code_11?????????
G
STP_Code_1_Weight2?/
inputs/STP_Code_1_Weight?????????
;
STP_Code_21,?)
inputs/STP_Code_21?????????
G
STP_Code_2_Weight2?/
inputs/STP_Code_2_Weight?????????
1
SixT_2'?$
inputs/SixT_2?????????	
-
Year%?"
inputs/Year?????????
p

 
? "???????????
$__inference_signature_wrapper_130910?F~?????????????????????????????? !*+45???
? 
???
B
Application_Area_1,?)
Application_Area_1?????????
P
Application_Area_1_Weight3?0
Application_Area_1_Weight?????????
B
Application_Area_2,?)
Application_Area_2?????????
P
Application_Area_2_Weight3?0
Application_Area_2_Weight?????????
8
Cowork_Abroad'?$
Cowork_Abroad?????????
2

Cowork_Cor$?!

Cowork_Cor?????????
4
Cowork_Inst%?"
Cowork_Inst?????????
2

Cowork_Uni$?!

Cowork_Uni?????????
2

Cowork_etc$?!

Cowork_etc?????????
4
Econ_Social%?"
Econ_Social?????????	
2

Green_Tech$?!

Green_Tech?????????	
6
Log_Duration&?#
Log_Duration?????????
6
Log_RnD_Fund&?#
Log_RnD_Fund?????????
2

Multi_Year$?!

Multi_Year?????????	
6
N_Patent_App&?#
N_Patent_App?????????
6
N_Patent_Reg&?#
N_Patent_Reg?????????
B
N_of_Korean_Patent,?)
N_of_Korean_Patent?????????
2

N_of_Paper$?!

N_of_Paper?????????
.
N_of_SCI"?
N_of_SCI?????????
D
National_Strategy_2-?*
National_Strategy_2?????????	
,
RnD_Org!?
RnD_Org?????????	
0
	RnD_Stage#? 
	RnD_Stage?????????	
4
STP_Code_11%?"
STP_Code_11?????????
@
STP_Code_1_Weight+?(
STP_Code_1_Weight?????????
4
STP_Code_21%?"
STP_Code_21?????????
@
STP_Code_2_Weight+?(
STP_Code_2_Weight?????????
*
SixT_2 ?
SixT_2?????????	
&
Year?
Year?????????"3?0
.
output_1"?
output_1?????????