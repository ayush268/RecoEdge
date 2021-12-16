package org.nimbleedge.recoedge.models

import java.security.{MessageDigest => MD}
import java.nio.ByteBuffer

abstract class Identifier {
    def computeDigest() : Array[Byte]
    def toList(): List[Identifier] = List.empty
    val digest : Array[Byte] = computeDigest()
    override val hashCode : Int = ByteBuffer.wrap(digest.slice(0,4)).getInt

    def hash(baseName: String, args: List[Identifier]): Array[Byte] = {
        val md = MD.getInstance("SHA-256")
        md.reset()
        md.update(baseName.getBytes("UTF-8"))
        args.foreach(
            arg => md.update(arg.digest)
        )
        md.digest()
    }

    override def equals(identifier_val: Any): Boolean = {
        identifier_val match {
            case i : Identifier => hashCode == identifier_val.hashCode && MD.isEqual(digest, i.digest)
            case _ => false
        }
    }
}

case class OrchestratorIdentifier(id: String) extends Identifier {
    // String Representation
    override def toString(): String = id

    // Get List
    override def toList(): List[Identifier] = List(this)

    // Hash digest
    override def computeDigest(): Array[Byte] = hash("_Orc" + id, List.empty)
}

case class AggregatorIdentifier(parentIdentifier: Identifier, id: String) extends Identifier {
    // String Representation
    override def toString(): String = parentIdentifier.toString() + " -> " + id

    // Get List
    override def toList(): List[Identifier] = parentIdentifier.toList().appended(this)

    // Hash digest
    override def computeDigest(): Array[Byte] = hash("_Agg" + id, parentIdentifier.toList())
}

case class TrainerIdentifier(parentIdentifier: Identifier, id: String) extends Identifier {
    // String Representation
    override def toString(): String = parentIdentifier.toString() + " -> " + id

    // Get List
    override def toList(): List[Identifier] = parentIdentifier.toList().appended(this)

    // Hash digest
    override def computeDigest(): Array[Byte] = hash("_Tra" + id, parentIdentifier.toList())
}