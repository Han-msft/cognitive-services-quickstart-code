
using Azure;
using Azure.AI.Vision.Face;

// URL path for the images.
const string IMAGE_BASE_URL = "https://raw.githubusercontent.com/Azure-Samples/cognitive-services-sample-data-files/master/Face/images/";

// From your Face subscription in the Azure portal, get your subscription key and endpoint.
var endpoint = Environment.GetEnvironmentVariable("VISION_ENDPOINT") ?? throw new Exception("Missing VISION_ENDPOINT environment variable.");
var key = Environment.GetEnvironmentVariable("VISION_KEY") ?? throw new Exception("Missing VISION_KEY environment variable.");

var personDictionary = new Dictionary<string, string[]>{
    { "Family1-Dad", new[] { "Family1-Dad1.jpg", "Family1-Dad2.jpg" } },
    { "Family1-Mom", new[] { "Family1-Mom1.jpg", "Family1-Mom2.jpg" } },
    { "Family1-Son", new[] { "Family1-Son1.jpg", "Family1-Son2.jpg" } },
    { "Family1-Daughter", new[] { "Family1-Daughter1.jpg", "Family1-Daughter2.jpg" } },
    { "Family2-Lady", new[] { "Family2-Lady1.jpg", "Family2-Lady2.jpg" } },
    { "Family2-Man", new[] { "Family2-Man1.jpg", "Family2-Man2.jpg" } }
};

var administrationClient = new FaceAdministrationClient(new Uri(endpoint), new AzureKeyCredential(key));
var client = new FaceClient(new Uri(endpoint), new AzureKeyCredential(key));

// Recognition model 4 was released in 2021 February.
// It is recommended since its accuracy is improved
// on faces wearing masks compared with model 3,
// and its overall accuracy is improved compared
// with models 1 and 2.
var recognitionModel = FaceRecognitionModel.Recognition04;

// Create a new PersonGroup
var personGroupId = "my-person-group";
Console.WriteLine($"Creating a new person group: {personGroupId}");
await administrationClient.CreatePersonGroupAsync(personGroupId, "My Person Group", recognitionModel: recognitionModel);
var personIdToPerson = new Dictionary<Guid, string>();

// The similar faces will be grouped into a single person group person.
foreach (var (personName, images) in personDictionary)
{
    // Limit TPS
    await Task.Delay(250);

    // Create a new Person
    Console.WriteLine($"Creating a new person: {personName}");
    var createPersonResponse = await administrationClient.CreatePersonGroupPersonAsync(personGroupId, personName);
    var personId = createPersonResponse.Value.PersonId;
    personIdToPerson[personId] = personName;

    foreach (var image in images)
    {
        var imageUri = new Uri($"{IMAGE_BASE_URL}{image}");
        Console.WriteLine($"Check if the image {image} is of sufficient quality for recognition");
        var detectResponse = await client.DetectFromUrlAsync(
            imageUri,
            FaceDetectionModel.Detection03,
            recognitionModel,
            returnFaceId: false,
            returnFaceAttributes: [FaceAttributeType.Recognition04.QualityForRecognition]);

        var sufficientQuality = detectResponse.Value.All(face => face.FaceAttributes.QualityForRecognition == QualityForRecognition.High);
        if (!sufficientQuality)
        {
            Console.WriteLine($"The image {image} is not of sufficient quality for recognition");
            continue;
        }

        // Add a face to the Person
        Console.WriteLine($"Adding face to {personName} from image: {image}");
        var face = await administrationClient.AddPersonGroupPersonFaceFromUrlAsync(personGroupId, personId, imageUri, userData: image);
    }
}

// Train the PersonGroup
Console.WriteLine("Training the person group");
await administrationClient.TrainPersonGroupAsync(WaitUntil.Completed, personGroupId);

// A group photo that includes some of the persons you seek to identify from your dictionary.
var sourceFaceImage = new Uri($"{IMAGE_BASE_URL}identification1.jpg");
Console.WriteLine();
Console.WriteLine("Detecting faces in source image");
var detectSourceResponse = await client.DetectFromUrlAsync(
    sourceFaceImage,
    FaceDetectionModel.Detection03,
    recognitionModel,
    returnFaceId: true,
    returnFaceAttributes: [FaceAttributeType.Recognition04.QualityForRecognition]);
var sourceFaceIdWithSufficientQuality = detectSourceResponse.Value
    .Where(detectedFace => detectedFace.FaceAttributes.QualityForRecognition == QualityForRecognition.High)
    .Select(detectedFace => detectedFace.FaceId.Value);

// Identify the faces in the person group
Console.WriteLine("Identifying faces in source image");
var identifyResponse = await client.IdentifyFromPersonGroupAsync(sourceFaceIdWithSufficientQuality, personGroupId);

foreach (var identifiedFace in identifyResponse.Value)
{
    if (!identifiedFace.Candidates.Any())
    {
        Console.WriteLine($"No person identified for {identifiedFace.FaceId}");
    }
    else
    {
        foreach (var candidate in identifiedFace.Candidates)
        {
            Console.WriteLine($"Person {personIdToPerson[candidate.PersonId]} is identified for {identifiedFace.FaceId} with a confidence of {candidate.Confidence}");
        }
    }
}

// Verify can be used to check if face belong to the specific person.
var targetFaceId = sourceFaceIdWithSufficientQuality.First();
var dadPersonId = personIdToPerson.Where(pair => pair.Value == "Family1-Dad").Select(pair => pair.Key).First();
Console.WriteLine();
Console.WriteLine($"Verifying face {targetFaceId} with the person {dadPersonId}");
var verifyResponse = await client.VerifyFromPersonGroupAsync(targetFaceId, personGroupId, dadPersonId);
Console.WriteLine($"The confidence of the face belonging to Family1-Dad is: {verifyResponse.Value.Confidence}");

// Delete the PersonGroup
Console.WriteLine();
Console.WriteLine("Deleting the person group");
await administrationClient.DeletePersonGroupAsync(personGroupId);
