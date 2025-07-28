import * as cdk from "aws-cdk-lib";
import { Construct } from "constructs";
import * as kvs from "aws-cdk-lib/aws-kinesisvideo";

export class KvsStack extends cdk.Stack {
  constructor(scope: Construct, id: string, props?: cdk.StackProps) {
    super(scope, id, props);

    new kvs.CfnStream(this, "OddKivaTestVideoStream", {
      name: "oddkiva-test-video-stream",
      deviceName: "webcam",
      dataRetentionInHours: 24,
    });
  }
}

