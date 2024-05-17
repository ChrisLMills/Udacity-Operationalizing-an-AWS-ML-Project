# Udacity-Operationalizing-an-AWS-ML-Project-

# Step 1 – Train and deploy model on SageMaker Notebook instance 

![Notebook instance](https://github.com/ChrisLMills/Udacity-Operationalizing-an-AWS-ML-Project-/assets/31799634/e848857c-4815-42b3-9ea0-5dfed818bd90)
<ins>Notebook instance

![S3 Bucket](https://github.com/ChrisLMills/Udacity-Operationalizing-an-AWS-ML-Project-/assets/31799634/9190b060-c5aa-4c03-94dd-80d07d056ce6)
<ins>S3 Bucket

I chose to begin with a ml.m5.xlarge instance (2 cores) as a baseline against which to make further instance selections based on its performance. 

I chose a m5 instance rather than a t3 as my CPU requirements were going to be constant. 
We see in the below metrics that I was utilising full CPU capacity was at its maximum, so we could consider going for a higher instance.

![Tuner jobs](https://github.com/ChrisLMills/Udacity-Operationalizing-an-AWS-ML-Project-/assets/31799634/5a57bde6-8d7d-4f27-a692-c39fcc354dd2)
<ins>Tuner jobs

From the utilisation metrics across both parallel training jobs, we can see that near 200% CPU capacity was being used (100% x 2 cores), so we could consider increasing to a ml.m5.2xlarge which has 4 cores.

![Resource utilization for tuner job 1](https://github.com/ChrisLMills/Udacity-Operationalizing-an-AWS-ML-Project-/assets/31799634/64ef6ce3-059c-41de-b3d0-a5950007d393)
<ins>Resource utilization for tuner job 1

![Resource utilization for tuner job 2](https://github.com/ChrisLMills/Udacity-Operationalizing-an-AWS-ML-Project-/assets/31799634/a9fe5416-99eb-4a91-8c85-bacc0e55d775)
<ins>Resource utilization for tuner job 2

Alternatively, we could try a ml.p3.2xlarge instance to see how adding 1 GPU compares.
For the model fit training job, I used ml.m5.2xlarge and got 365% usage for the 4 cores, so this seems like a better option than the ml.m5.xlarge instance. 

![Resource utilization for model training job](https://github.com/ChrisLMills/Udacity-Operationalizing-an-AWS-ML-Project-/assets/31799634/e56ec8a5-cfa9-48c4-b3dd-88c81d132fef)
<ins>Resource utilization for model training job

For my multi-instance training, I dropped the instance type to ml.m5.xlarge and increased the number of instances to 2. This will provide a comparison between the instance cost and time saving to see which of the two is more efficient. 

Costs:
ml.m5.xlarge: 0.1920
ml.m5.2xlarge: 0.3840

![Model training job](https://github.com/ChrisLMills/Udacity-Operationalizing-an-AWS-ML-Project-/assets/31799634/ee14e0dc-d69e-4c79-bb1f-1dcdf56a3cf0)
<ins>Single instance ml.m5.2xlarge training job

![Multi-instance model training job](https://github.com/ChrisLMills/Udacity-Operationalizing-an-AWS-ML-Project-/assets/31799634/93a4769e-730d-4c8d-8385-653bca0f1a3a)
<ins>Multi-instance ml.m5.xlarge training job

The result is that it is more cost effective to use the single instance of ml.m5.2xlarge than 2 instances of ml.m5.xlarge.

![Deployed endpoint](https://github.com/ChrisLMills/Udacity-Operationalizing-an-AWS-ML-Project-/assets/31799634/aa10cbe6-a1bf-4c3f-b1fe-6629336d8577)
<ins>Deployed endpoint

# Step 2 – train model on EC2 instance 

I chose a g4dn.xlarge instance to be able to run the DeepLearning AMI with OSS Nvidia drivers. I found this out the long way, by trying to use this AMI with m5 instances, and realising that I could not activate a PyTorch environment when I did. 

![EC2 instance](https://github.com/ChrisLMills/Udacity-Operationalizing-an-AWS-ML-Project-/assets/31799634/1b143d4e-1269-45a2-b351-20fcafd5fd5f)
<ins>EC2 instance

![EC2 trained model](https://github.com/ChrisLMills/Udacity-Operationalizing-an-AWS-ML-Project-/assets/31799634/07ad8358-288e-4b82-a645-313b03d87ac0)
<ins>EC2 trained model

Differences between the script used in the SageMaker Notebook and the one used in the EC2 instance include:

•	There is no logger in the EC2 script
•	Args are not passed via the estimator’s hyperparameter object
•	There is no __Main__ function in the EC2 script

# Step 3 – Lambda script

To invoke the deployed endpoint, a boto3 runtime is required. To this, the endpoint name and event object is passed. This passes the object to the endpoint and returns a result. This result, taken from the JSON body, is included in the return statement. 

![Lambda setup](https://github.com/ChrisLMills/Udacity-Operationalizing-an-AWS-ML-Project-/assets/31799634/37014298-0c5e-443b-89f1-98e19fbebc7a)
<ins>Lambda setup

# Step 4 – Lambda permission

```
{
  "statusCode": 200,
  "headers": {
    "Content-Type": "text/plain",
    "Access-Control-Allow-Origin": "*"
  },
  "type-result": "<class 'str'>",
  "COntent-Type-In": "LambdaContext([aws_request_id=7809bce2-6b43-4256-9efa-03e9499b1331,log_group_name=/aws/lambda/Endpoint-function,log_stream_name=2024/05/16/[$LATEST]a4a959ef164e4fc393910807308d8f7a,function_name=Endpoint-function,memory_limit_in_mb=128,function_version=$LATEST,invoked_function_arn=arn:aws:lambda:us-east-1:992617474323:function:Endpoint-function,client_context=None,identity=CognitoIdentity([cognito_identity_id=None,cognito_identity_pool_id=None])])",
  "body": "[[-6.933251857757568, -4.115185260772705, -1.1860870122909546, -0.051640577614307404, -1.8509540557861328, -4.516417503356934, -0.6620293259620667, -2.1556873321533203, -6.661304473876953, -1.7848674058914185, -0.39824116230010986, -1.9597395658493042, -0.8100516200065613, 0.15559150278568268, -3.832732915878296, -1.5320160388946533, -8.546006202697754, -3.748659372329712, -5.5662713050842285, 0.28723448514938354, -3.8938241004943848, -0.8271301984786987, -7.9896135330200195, -6.753727436065674, -5.102131366729736, -8.316194534301758, -1.0639723539352417, -4.34860897064209, -3.351830244064331, -3.5733888149261475, -1.8442684412002563, -6.013882160186768, -7.711957931518555, -5.1997294425964355, -7.592862606048584, -4.665738105773926, -3.211360454559326, -2.893667221069336, -2.3338818550109863, -2.738603115081787, -2.6702356338500977, -2.386967897415161, 1.1263642311096191, -3.8626484870910645, 1.1946799755096436, -8.156968116760254, -2.7372381687164307, -1.9260501861572266, -3.834108829498291, -1.6010427474975586, -5.220654487609863, -7.219725608825684, -6.7797064781188965, -1.4722788333892822, -4.1217875480651855, -1.1247429847717285, -6.003917217254639, -5.570618629455566, -2.945807695388794, -3.6547067165374756, -5.10927677154541, -4.489850044250488, -7.5780205726623535, -7.336756706237793, -4.498852729797363, -7.307641983032227, -2.9481451511383057, -7.022530555725098, -3.7528958320617676, -1.7472091913223267, -0.31721630692481995, -5.75593900680542, -7.086024284362793, -7.735294342041016, -5.840084552764893, -0.9356353878974915, -8.872957229614258, -3.112175703048706, -5.91922664642334, -3.830321788787842, 0.3538955748081207, -6.558791637420654, 0.6257588863372803, -0.1014268696308136, -5.897650718688965, -4.0747809410095215, -2.5116963386535645, -4.1817946434021, -2.5635459423065186, -0.3762665092945099, -6.910386562347412, -6.645389556884766, -5.7688775062561035, -6.769272327423096, -5.362224578857422, -1.839355230331421, -2.6273438930511475, -5.6136555671691895, -5.773927211761475, -4.775604724884033, -9.654525756835938, -4.876921653747559, -5.716403484344482, -4.9359588623046875, -6.929784297943115, -6.931429862976074, -2.7079906463623047, -0.4363161623477936, -3.5302560329437256, 0.2724008560180664, -2.6324799060821533, -2.016328811645508, -7.4810099601745605, -4.188353538513184, -4.309124946594238, -1.040969729423523, -6.541793346405029, -0.2283555120229721, -4.641557693481445, 0.3935657739639282, -1.7111499309539795, -3.8666062355041504, -4.364243507385254, -1.3981400728225708, -8.266094207763672, -2.4975385665893555, -2.453230857849121, 0.2529045343399048, -7.871884822845459, -5.786275386810303, -5.272493362426758, 0.05237623304128647, -4.389797687530518]]"
}
```

![IAM roles](https://github.com/ChrisLMills/Udacity-Operationalizing-an-AWS-ML-Project-/assets/31799634/659d65a0-e6dc-4147-99c0-25efdcf081eb)
<ins>IAM roles

There are a number of AWS Service Roles which appear to be unnecessary but allow access to storage and compute resources. If someone hacked into this account, they would have access to a lot of resources, at the organisation’s expense. 

I added a SageMaker All Access policy to the Lambda function role. However, this could be further stripped down to be made to only handle the endpoint. 

# Step 5 – Concurrency and Auto-scaling

I chose to provision 10 instances for my lambda function concurrency and 3 instances for my endpoint auto-scaling. These choices were based on the low latency of the inference which I assume will be able to handle a larger volume than what a single Lambda function will be able to handle, given that it cannot handle multiple calls simultaneously. 
The reason I did not reserve any concurrency for my lambda function is because of its cost, and that I do not yet have any data on the expected throughput. 

![Lambda concurrency](https://github.com/ChrisLMills/Udacity-Operationalizing-an-AWS-ML-Project-/assets/31799634/45992cd8-7ef1-4ec5-9399-869c129231b9)
<ins>Lambda concurrency

![Endpoint auto-scaling](https://github.com/ChrisLMills/Udacity-Operationalizing-an-AWS-ML-Project-/assets/31799634/eece5d47-c994-4d0c-bac1-2a09380e737c)
<ins>Endpoint auto-scaling

